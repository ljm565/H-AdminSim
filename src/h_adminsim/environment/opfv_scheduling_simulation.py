import re
import os
import json
import random
from copy import deepcopy
from importlib import resources
from patientsim import PatientAgent
from decimal import Decimal, getcontext
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from typing import Tuple, Union, Optional, TYPE_CHECKING

from h_adminsim.registry.errors import (
    SchedulingError,
    ToolCallingError, 
    DataNotFoundError, 
)
from h_adminsim.registry import (
    STATUS_CODES,
    SCHEDULE_STATUS,
    OPFV_PREFERENCE_PHRASE_STAFF,
    OPFV_PREFERENCE_PHRASE_PATIENT, 
)
from h_adminsim.tools.callback import TokenUsageCallback
from h_adminsim.tools.sanity_checker import SanityChecker
from h_adminsim.tools import SchedulingRule, scheduling_tool_calling
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.common_utils import *

if TYPE_CHECKING:
    from h_adminsim.pipeline import HospitalMAS
    from h_adminsim.agent import SchedulingAdminStaffAgent
    from h_adminsim.environment.hospital import HospitalEnvironment



class OPFVSchedulingSimulation:
    def __init__(self,
                 patient_agent: PatientAgent,
                 admin_staff_mas: "HospitalMAS",
                 metadata: dict,
                 department_data: dict,
                 environment: "HospitalEnvironment",
                 scheduling_strategy: str = 'tool_calling',
                 preference_rejection_prob: float = 0.3,
                 preference_rejection_prob_decay: float = 0.5,
                 fhir_integration: bool = False,
                 schedule_rejection_prompt_path: Optional[str] = None,
                 sanity_checker: Optional[SanityChecker] = None):
        
        # Initialize simulation parameters
        getcontext().prec = 10
        self.patient_agent = patient_agent
        self.admin_staff_mas = admin_staff_mas
        self.environment = environment
        self._START_HOUR = metadata['time']['start_hour']
        self._END_HOUR = metadata['time']['end_hour']
        self._TIME_UNIT = metadata['time']['interval_hour']
        self._DAY = metadata['days']
        self.scheduling_strategy = scheduling_strategy
        self.preference_rejection_prob = preference_rejection_prob
        self.preference_rejection_prob_decay = preference_rejection_prob_decay
        self.fhir_integration = fhir_integration
        self._init_prompt(schedule_rejection_prompt_path)
        self.sanity_checker = sanity_checker
        self.rules = SchedulingRule(metadata, department_data, self.environment, self.fhir_integration)


    @property
    def scheduling_agent(self) -> "SchedulingAdminStaffAgent":
        """
        The scheduling worker (a leaf of the MAS tree).

        Scheduling, cancellation, and rescheduling are all handled by the same
        ``'first_visit_scheduling'`` worker and are independent of the intake flow,
        so it is fetched from the MAS on demand rather than cached as instance state.
        """
        return self.admin_staff_mas.get_agent('first_visit_scheduling')

    
    def _init_prompt(self, schedule_rejection_prompt_path: Optional[str] = None):
        """
        Initialize the schedule rejection system prompt for the administration staff agent.

        Args:
            schedule_rejection_prompt_path (Optional[str], optional): Path to a custom schedule rejection system prompt file. 
                                                                      If not provided, the default system prompt will be used. Defaults to None.

        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.
        """
        # Initialilze with the default system prompt
        if not schedule_rejection_prompt_path:
            prompt_file_name = "opfv_schedule_patient_rejected_system.txt"
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            self.rejection_system_prompt_template = file_path.read_text()
        
        # User can specify a custom system prompt
        else:
            if not os.path.exists(schedule_rejection_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {schedule_rejection_prompt_path}"))
            with open(schedule_rejection_prompt_path, 'r') as f:
                self.rejection_system_prompt_template = f.read()

        # Additional prompts for streaming scheduling simulation
        self.patient_satisfaction_system_prompt = (
            "You are a patient looking to schedule an appointment. "
            "Assume that the latest schedule proposed by the hospital administrative staff is satisfactory."
        )
        self.natural_end_phrase = (
            "{schedule}\n"
            "Respond to this suggested schedule, ending with a thank-you nuance.\n"
            "- If the conversation history shows that the same schedule was previously proposed and you rejected it, "
            "respond with resigned understanding that this is likely the earliest available "
            "(e.g., 'Oh, this must be the earliest available. Alright, thank you.'), keeping it under 15 words.\n"
            "- Otherwise, respond with plain satisfaction in 5 words or fewer, "
            "conveying only satisfaction and no dissatisfaction."
        )
        self.patient_evaluation_system_prompt = (
            "You are a patient evaluating whether the proposed appointment meets your scheduling preference. "
            "Your preference is for a specific {preference}: {preferred_condition}."
        )
        self.patient_schedule_evaluation_phrase = (
            "{schedule}\n"
            "Evaluate whether this appointment meets your {preference} preference ({preferred_condition}).\n"
            "- If your preference is 'doctor': accept if the appointment is with {preferred_condition}.\n"
            "- If your preference is 'date': accept if the appointment date is on or after {preferred_condition}.\n"
            "If acceptable, respond with brief acceptance (5 words or fewer) ending with a thank-you, "
            "and include '#ACCEPT' at the very end. "
            "Otherwise, briefly express dissatisfaction and state what you need instead."
        )
        self.end_phrase = "Thank you."


    def _init_agents(self, verbose: bool = True):
        """
        Reset the conversation histories and token usage records of both the Patient and Doctor agents.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        """
        self.patient_agent.reset_history(verbose=verbose)
        self.admin_staff_mas.reset(verbose=verbose)


    def _init_history(self):
        """
        Reset the dialogue histories.
        """
        self.dialog_history = {
            'scheduling': [],
            'cancel': [],
            'reschedule': [],
        }

    
    def _to_lc_history(self, key: str) -> list:
        """
        Convert the dialog history for the given key into LangChain message objects.

        Args:
            key (str): Key identifying which dialog history to convert.

        Returns:
            list: A list of LangChain HumanMessage and AIMessage objects.
        """
        msgs = []
        for m in self.dialog_history[key]:
            if m["role"] == "Patient":
                msgs.append(HumanMessage(content=m["content"]))
            elif m["role"] == "Staff":
                msgs.append(AIMessage(content=m["content"]))
        return msgs


    def _render_staff_reply(self,
                            staff_response: dict,
                            reply_type: str,
                            natural_express: bool = True) -> str:
        """
        Turn a structured staff scheduling result into a natural-language utterance.

        Args:
            staff_response (dict): The result of `scheduling`, either a clarification (``type == 'text'``) or a schedule proposal (``type == 'tool'``).
            reply_type (str): Reply types of the scheduling agent.
            natural_express (bool, optional): Whether to phrase a schedule proposal naturally instead of dumping the raw schedule dict. Defaults to True.

        Returns:
            str: The staff utterance to show the patient.
        """
        # Clarification message
        if staff_response['type'] == 'text':
            return staff_response['result']

        # Tool calling result
        elif staff_response['type'] == 'tool':
            if reply_type == 'scheduling':
                pred_schedule = staff_response['result']
            
                # Response formatting
                try:
                    if natural_express:
                        _schedule = pred_schedule['schedule']
                        _doctor = list(_schedule.keys())[0]
                        _date, _st, _tr = _schedule[_doctor]['date'], _schedule[_doctor]['start'], _schedule[_doctor]['end']
                        _format = random.choice(self.scheduling_agent.natural_schedule_suggestion) \
                            if isinstance(self.scheduling_agent.natural_schedule_suggestion, list) \
                                else self.scheduling_agent.natural_schedule_suggestion
                        return _format.format(doctor=_doctor, date=_date, start=_st, end=_tr)
                    return self.scheduling_agent.schedule_suggestion.format(schedule=pred_schedule)
                except:
                    try:
                        return self.scheduling_agent.schedule_suggestion.format(schedule=pred_schedule)
                    except:
                        return str(pred_schedule)

            elif reply_type == 'cancel':
                result = staff_response['result']

                # A wrong identification cancels nothing; the loop surfaces it as a failure.
                if result['cancelled_schedule'] is None:
                    return ""

                # Successful cancellation
                cancelled_schedule = {k: v for k, v in result['cancelled_schedule'].items()
                                      if k in ['patient', 'attending_physician', 'department', 'date', 'schedule']}
                return f"I've cancelled this schedule: {cancelled_schedule}"

            elif reply_type == 'reschedule':
                tmp_flag = staff_response.get('tmp_flag')
                result = staff_response['result']

                # Failure cases (identification / scheduling) have nothing to confirm;
                # the loop surfaces them as raises, so render nothing here.
                if tmp_flag not in ('waiting_list', 'reschedule'):
                    return ""

                original = {k: v for k, v in result['original_schedule'].items()
                            if k in ['patient', 'attending_physician', 'department', 'date', 'schedule']}

                # No slot available -> added to the waiting list
                if tmp_flag == 'waiting_list':
                    return f"There are no available times. I've added this schedule to the waiting list: {original}"

                # Successfully moved to an earlier slot
                new = {k: v for k, v in result['new_schedule'].items()
                       if k in ['patient', 'attending_physician', 'department', 'date', 'schedule']}
                return f"I've moved your original schedule: {original} to the new one: {new}"


    def _accumulate_staff_tokens(self,
                                 staff_response: dict,
                                 staff_token_stats: dict,
                                 staff_token_callback: TokenUsageCallback) -> dict:
        """
        Merge this turn's staff token usage into the running stats and return them.

        Args:
            staff_response (dict): The staff scheduling result (carries ``'token'`` for reasoning).
            staff_token_stats (dict): The running per-key token usage to update.
            staff_token_callback (TokenUsageCallback): Cumulative callback for the tool-calling path.

        Returns:
            dict: The updated token statistics.
        """
        if self.scheduling_strategy == 'tool_calling':
            return staff_token_callback.token_usage
        for k, v in staff_response['token'].items():
            if k not in staff_token_stats:
                staff_token_stats[k] = deepcopy(v)
            else:
                staff_token_stats[k].extend(v)
        return staff_token_stats


    def _finish_scheduling_turn(self,
                                reply_type: str,
                                verbose: bool = False):
        """
        Hand the floor back to the orchestrator once the scheduling eval loop is done.

        Args:
            reply_type (str): Reply types of the scheduling agent.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        if len(self.admin_staff_mas.path) <= 1:
            return
        
        closing = self.dialog_history[reply_type][-1]['content']
        messages = self.admin_staff_mas.state.messages
        
        # When end abnormally
        if messages and messages[-1]['content'] == closing:
            return
        
        # When end normally
        self.admin_staff_mas.chat(
            user_prompt=closing,
            using_multi_turn=False,
            verbose=verbose,
            is_done=True,
        )


    @staticmethod
    def postprocessing(strategy: str,
                       data: Union[str, dict],
                       filtered_doctor_information: Optional[dict] = None) -> Union[str, dict]:
        """
        Attempts to parse the given text as JSON. If parsing succeeds, returns a dictionary;
        otherwise, returns the original string.

        Args:
            strategy (str): Scheduling strategy. It must be either `reasoning` or `tool_calling`.
            data (Union[str, dict]): The text output to post-process, potentially a JSON-formatted string. 
            filtered_doctor_information (Optional[dict], optional): Department-filtered doctor information 
                                                                    to postprocess the schedule by tool_calling strategy.

        Returns:
            Union[str, dict]: A dictionary if the text is valid JSON, otherwise the original string.
        """
        if strategy == 'reasoning':
            try:
                if isinstance(data, str):
                    match = re.search(r'```json\s*(\{.*?\})\s*```', data, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                        text_dict = json.loads(json_str)
                    else:
                        try:
                            text_dict = json.loads(data)
                        except:
                            return data
                else:
                    text_dict = data
                
                assert len(text_dict) == 1 and all(k in text_dict for k in ['schedule'])   # Basic sanity check
                key = list(text_dict['schedule'].keys())[0]
                text_dict['schedule'][key]['start'] = float(text_dict['schedule'][key]['start'])
                text_dict['schedule'][key]['end'] = float(text_dict['schedule'][key]['end'])
                text_dict['schedule'][key]['date'] = str(text_dict['schedule'][key]['date'])
                return text_dict
            
            except:
                return str(data)
        
        elif strategy == 'tool_calling':
            doctor = data['doctor'][0]
            duration = filtered_doctor_information['doctor'][doctor]['outpatient_duration']
            date, st_hour = iso_to_date(data['schedule'][0]), iso_to_hour(data['schedule'][0])
            tr_hour = float(Decimal(str(duration)) + Decimal(str(st_hour)))
            return {'schedule': {doctor: {'date': date, 'start': st_hour, 'end': tr_hour}}}


    def update_patient_system_prompt(self, 
                                     patient_condition: Optional[dict] = None,
                                     rejected_preference: Optional[str] = None,
                                     new_system_prompt: Optional[str] = None):
        """
        Update a system prompt of the patient agent for proposed schedule rejection scenario etc.

        Args:
            patient_condition (Optional[dict], optional): Patient ground-truth condition including current preference.
            rejected_preference (Optional[str], optional): The scheduling preference proposed by the staff agent in the previous turn
                                                           that the patient must explicitly reject.
            new_system_prompt (Optional[str], optional): New system prompt to be updated.
        """
        if patient_condition is not None and rejected_preference is not None:
        # Build new system prompts for rejection scenario
            preference = patient_condition.get('preference')
            preference_desc = OPFV_PREFERENCE_PHRASE_PATIENT[preference] if preference != 'date' \
                    else OPFV_PREFERENCE_PHRASE_PATIENT[preference].format(date=patient_condition.get('valid_from'))
            rejected_preference_desc = OPFV_PREFERENCE_PHRASE_STAFF[rejected_preference] if rejected_preference != 'date' \
                    else OPFV_PREFERENCE_PHRASE_STAFF[rejected_preference].format(date='a specific date')    
            system_prompt = self.rejection_system_prompt_template.format(
                preference=preference,
                preference_desc=preference_desc,
                preferred_doctor=patient_condition['preferred_doctor'],
                rejected_preference=rejected_preference_desc,
                personality=self.patient_agent.personality,
            )
        else:
            system_prompt = new_system_prompt

        # Update new system prompts for rejection scenario
        self.patient_agent.system_prompt = system_prompt
        if len(self.patient_agent.client.histories) and \
            isinstance(self.patient_agent.client.histories[0], dict) and \
                self.patient_agent.client.histories[0].get('role') == 'system':
            self.patient_agent.client.histories[0]['content'][0]['text'] = system_prompt


    def _get_rescheduled_result(self,
                                known_condition: dict,
                                doctor_information: Optional[dict] = None,
                                **kwargs) -> dict:
        """
        Reschedule with the only scheduling tools.

        Args:
            known_condition (dict): Patient conditions known to the staff.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details. Defaults to None.

        Returns:
            dict: Rescheduled schedule.
        """
        # Sanity check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        filtered_doctor_information = self.environment.get_doctor_schedule(
            doctor_information=doctor_information,
            department=known_condition['department'],
            fhir_integration=self.fhir_integration and doctor_information is None,
        )
        _schedule_client = self.scheduling_agent.build_agent(
            rule=self.rules, 
            doctor_info=filtered_doctor_information,
            only_schedule_tool=True
        )
        new_schedule = self.scheduling(
            client=_schedule_client,
            known_condition=known_condition,
            doctor_information=doctor_information,
            reschedule_flag=True,
            **kwargs
        )['result']
        return new_schedule
    
    
    def _check_reschedule_validity(self,
                                   idx: int,
                                   new_schedule: dict,
                                   original_schedule: dict,
                                   doctor_information: dict) -> Optional[dict]:
        """
        Check the rescheduling availability.

        Args:
            idx (int): Index of the requested schedule (original schedule index).
            new_schedule (dict): New earliest schedule available.
            original_schedule (dict): The original schedule.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details.

        Returns:
            Optional[dict]: New schedule if the rescheduling available; otherwise None.
        """
        pred_doctor_name = list(new_schedule['schedule'].keys())[0]
        old_iso_time = get_iso_time(original_schedule['schedule'][0], original_schedule['date'])
        new_iso_time = get_iso_time(new_schedule['schedule'][pred_doctor_name]['start'], new_schedule['schedule'][pred_doctor_name]['date'])
        if compare_iso_time(old_iso_time, new_iso_time):
            self.rules.cancel_schedule(idx, doctor_information, original_schedule)
            final_schedule = {
                'visit_type': 'first_visit',
                'patient': original_schedule['patient'],
                'attending_physician': pred_doctor_name,
                'department': original_schedule['department'],
                'date': new_schedule['schedule'][pred_doctor_name]['date'],
                'schedule': [
                    new_schedule['schedule'][pred_doctor_name]['start'], 
                    new_schedule['schedule'][pred_doctor_name]['end']
                ],
                'patient_intention': original_schedule['patient_intention'],
                'preference': original_schedule.get('preference'),
                'preferred_doctor': original_schedule.get('preferred_doctor'),
                'valid_from': original_schedule.get('valid_from'),
                'test': None,
                'last_updated_time': self.environment.current_time
            }
            return final_schedule
        return None
    

    def _make_reschedule_pipeline(self, doctor_information: Optional[dict] = None, **kwargs):
        """
        Build a callable that runs the post-retrieval rescheduling pipeline:
        _get_rescheduled_result -> sanity check -> _check_reschedule_validity -> waiting list.

        Args:
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s).
            **kwargs: Additional keyword arguments forwarded to the inner scheduling agent.

        Returns:
            Callable[[int, dict], dict]: A pipeline function returning a dict with keys
                'action' ('reschedule' | 'waiting_list' | 'schedule_fail'),
                'new_schedule' (dict | None), and optional 'status_code'.
        """
        def pipeline(idx: int, original_schedule: dict) -> dict:
            try:
                new_schedule = self._get_rescheduled_result(
                    known_condition=original_schedule,
                    doctor_information=doctor_information,
                    **kwargs,
                )
            except Exception:
                return {'action': 'schedule_fail', 'new_schedule': None,
                        'status_code': STATUS_CODES['format']}

            if self.sanity_checker is not None:
                ok, code = self.sanity_checker.schedule_check(
                    prediction=new_schedule,
                    gt_patient_condition=original_schedule,
                    doctor_information=doctor_information,
                    environment=self.environment,
                )
                if not ok:
                    return {'action': 'schedule_fail', 'new_schedule': new_schedule,
                            'status_code': code}

            try:
                final = self._check_reschedule_validity(
                    idx=idx,
                    new_schedule=new_schedule,
                    original_schedule=original_schedule,
                    doctor_information=doctor_information,
                )
                if final is not None:
                    return {'action': 'reschedule', 'new_schedule': final, 'status_code': None}
                self.environment.add_waiting_list(idx, True)
                return {'action': 'waiting_list', 'new_schedule': None, 'status_code': None}
            except Exception:
                return {'action': 'schedule_fail', 'new_schedule': new_schedule,
                        'status_code': STATUS_CODES['format']}

        return pipeline


    def automatic_waiting_list_update(self,
                                      doctor_information: dict,
                                      **kwargs):
        """
        Update waiting list availability automatically.

        Args:
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details.

        Yields:
            dict: Updated (or not updated) doctor information and a result dictionary.
        """
        for turn, (idx, original) in enumerate(self.environment.waiting_list):
            if original['status'] == SCHEDULE_STATUS['scheduled']:
                new_schedule = self._get_rescheduled_result(
                    known_condition=original,
                    doctor_information=doctor_information,
                    **kwargs
                )

                # Sanity check
                ## No GT case
                if self.sanity_checker is None:
                    status, status_code = True, STATUS_CODES['correct']
                else:
                    status, status_code = self.sanity_checker.schedule_check(
                        prediction=new_schedule,
                        gt_patient_condition=original,
                        doctor_information=doctor_information,
                        environment=self.environment
                    )
                
                if status:
                    try:
                        final_schedule = self._check_reschedule_validity(
                            idx=idx,
                            new_schedule=new_schedule,
                            original_schedule=original,
                            doctor_information=doctor_information,
                        )
                        if final_schedule is not None:
                            result_dict = {
                                'gt': ['automatic rescheduling'],
                                'pred': [final_schedule],
                                'status': [True],
                                'status_code': [STATUS_CODES['correct']],
                                'dialog': ['automatic waiting list update from the system']
                            }
                            yield {'doctor_information': doctor_information, 'result_dict': result_dict, 'original': original}

                    except:
                        log('No sanity checker is available; an error occurred while parsing the prediction. Returning a failure result.', level='warning')
                        result_dict = {
                            'gt': ['automatic rescheduling'],
                            'pred': [new_schedule],
                            'status': [False],
                            'status_code': [STATUS_CODES['reschedule']['schedule'].format(status_code=STATUS_CODES['format'])],
                            'dialog': ['automatic waiting list update from the system']
                        }
                        yield {'doctor_information': doctor_information, 'result_dict': result_dict, 'original': original}

                else:
                    result_dict = {
                        'gt': ['automatic rescheduling'],
                        'pred': [new_schedule],
                        'status': [status],
                        'status_code': [STATUS_CODES['reschedule']['schedule'].format(status_code=status_code)],
                        'dialog': ['automatic waiting list update from the system']
                    }
                    yield {'doctor_information': doctor_information, 'result_dict': result_dict, 'original': original}


    def scheduling(self,
                   client: AgentExecutor,
                   known_condition: dict,
                   doctor_information: Optional[dict] = None, 
                   reschedule_flag: bool = False,
                   chat_history: list = [],
                   reasoning_max_tries: int = 0,
                   **kwargs) -> dict:
        """
        Make an appointment between the doctor and the patient.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            known_condition (dict): Patient conditions known to the staff.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details. Defaults to None.
            reschedule_flag (bool, optional): Whether this process is rescheduling or not. Defaults to False.
            chat_history (list, optional): Chat history. Defaults to [].

        Raises:
            ToolCallingError: If the agent fails to select or execute a valid scheduling tool.
            TypeError: If the prediction or inputs are of an unsupported type.

        Return
            dict: Scheduling processed result.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        # Initialization based on the known condition from the staff
        callback = kwargs.pop('callback', None)
        department = known_condition['department']
        filtered_doctor_information = self.environment.get_doctor_schedule(
            doctor_information=doctor_information,
            department=department,
            fhir_integration=self.fhir_integration and doctor_information is None,
        )
        
        # First, try to use the tool calling
        try:
            if self.scheduling_strategy != 'tool_calling':
                log('Scheduling strategy is set to `reasoning`, directly use the reasoning method.', level='warning')
                raise AssertionError
            
            # Invoke
            prediction = scheduling_tool_calling(
                client=client, 
                user_prompt=known_condition['patient_intention'],
                history=chat_history,
                callback=callback,
            )

            # Post-processing
            ## Scheduling result
            if prediction['type'] == 'tool':
                schedule = OPFVSchedulingSimulation.postprocessing(
                    strategy='tool_calling',
                    data=prediction['result'],
                    filtered_doctor_information=filtered_doctor_information,
                )
                prediction['result'] = schedule
            
            ## Dialogue
            elif prediction['type'] == 'text':
                if 'no tool' in prediction['result'].lower():
                    raise ToolCallingError(colorstr('red', 'Failed to choose an appropriate scheduling tool.'))
            
            ## Error
            else:
                raise TypeError(colorstr("red", "Error: Unexpected return type from scheduling method."))

        # If tool calling fails, fallback to LLM-based scheduling
        except Exception as e:
            if not isinstance(e, AssertionError):
                log(f'Exception occured: {e}', 'warning')
            
            if self.scheduling_strategy == 'tool_calling':
                log('Failed to select an appropriate tool. Falling back to reasoning-based scheduling.', level='warning')
            
            reschedule_desc = "Rescheduling requested. This is the rescheduling of a patient who wishes to move their appointment earlier due to a previous patient's cancelled reservation" \
                if reschedule_flag else 'Not requested.'
            filtered_doctor_information = self.environment.get_doctor_schedule(
                doctor_information=doctor_information,
                department=department,
                fhir_integration=self.fhir_integration and doctor_information is None,
                express_detail=True
            )
            current_time = f"{self.environment.current_time} (Date: {iso_to_date(self.environment.current_time)}, Time: {round(iso_to_hour(self.environment.current_time), 3)})"
            user_prompt = self.scheduling_agent.scheduling_user_prompt_template.format(
                START_HOUR=self._START_HOUR,
                END_HOUR=self._END_HOUR,
                TIME_UNIT=self._TIME_UNIT,
                CURRENT_TIME=current_time,
                DEPARTMENT=department,
                PREFERENCE=known_condition['patient_intention'], 
                RESCHEDULING_FLAG=reschedule_desc,
                DAY=self._DAY,
                DOCTOR=json.dumps(filtered_doctor_information, indent=2),
            )

            tries = 0
            while 1:
                schedule = self.scheduling_agent(
                    user_prompt,
                    using_multi_turn=False,
                    verbose=False,
                    **kwargs,
                )
                schedule = OPFVSchedulingSimulation.postprocessing(
                    strategy='reasoning',
                    data=schedule,
                )
                if isinstance(schedule, dict) or tries >= reasoning_max_tries:
                    break
                tries += 1

            if not isinstance(schedule, dict):
                self.scheduling_agent.reset_history(verbose=False)
                raise SchedulingError(colorstr('red', 'Reasoning fallback failed to produce a valid schedule JSON.'))

            prediction = {
                'type': 'tool',
                'result': schedule,
                'raw': None,
                'token': deepcopy(self.scheduling_agent.client.token_usages),
            }
            self.scheduling_agent.reset_history(verbose=False)

        return prediction
    

    def canceling(self, 
                  client: AgentExecutor,
                  patient_intention: str,
                  chat_history: list = []) -> dict:
        """
        Handle a multi-turn appointment cancellation request using a tool-calling agent.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            patient_intention (str): The patient's utterance expressing a rescheduling request.
            chat_history (list, optional): Chat history. Defaults to [].
        
        Raises:
            TypeError: If the prediction or inputs are of an unsupported type.

        Returns:
            dict: Cancelling processed result.
        """
        # Initialize
        result_dict = init_result_dict()

        # Invoke
        prediction = scheduling_tool_calling(
            client=client,
            user_prompt=patient_intention,
            history=chat_history,
        )
        
        # Canceling result
        if prediction['type'] == 'tool':
            if prediction['result']['status'] is None:
                # Schedule not found case: -> return: str
                if prediction['result']['index']['pred'] == -1:
                    prediction['type'] = 'text'
                    prediction['result'] = "Sorry, we couldn't find a matching appointment. Could you please check your appointment details again?"
                
                # No GT case
                else:
                    result_dict['gt'].append({'cancel': None})
                    result_dict['pred'].append({'cancel': prediction['result']['index']['pred']})
                    result_dict['status'].append(None)
                    result_dict['status_code'].append(None)
                    prediction['result_dict'] = result_dict

                return prediction
            
            # Retrieval fail case
            elif prediction['result']['status'] is False:
                result_dict['gt'].append({'cancel': prediction['result']['index']['gt']})
                result_dict['pred'].append({'cancel': prediction['result']['index']['pred']})
                result_dict['status'].append(prediction['result']['status'])
                result_dict['status_code'].append(STATUS_CODES['cancel']['identify'])
                prediction['result_dict'] = result_dict
                return prediction
            
            # Successful cancellation case -> return: dict
            else:
                result_dict['gt'].append({'cancel': prediction['result']['index']['gt']})
                result_dict['pred'].append({'cancel': prediction['result']['index']['pred']})
                result_dict['status'].append(prediction['result']['status'])
                result_dict['status_code'].append(STATUS_CODES['correct'])
                prediction['result_dict'] = result_dict
                return prediction

        # Clarification message case -> return: str
        elif prediction['type'] == 'text':
            return prediction

        # Error
        else:
            raise TypeError(colorstr("red", "Error: Unexpected return type from canceling method."))
    

    def rescheduling(self,
                     client: AgentExecutor,
                     patient_intention: str,
                     doctor_information: Optional[dict] = None,
                     chat_history: list = [],
                     **kwargs) -> dict:
        """
        Handle a multi-turn appointment rescheduling request using a tool-calling agent.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            patient_intention (str): The patient's utterance expressing a rescheduling request.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved,
                                                           including availability and other relevant details. Defaults to None.
            chat_history (list, optional): Chat history. Defaults to [].

        Raises:
            TypeError: If the returned type is not supported.

        Returns:
            dict: Rescheduling processed result.
        """
        # Sanity check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        # Initialize
        result_dict = init_result_dict()

        # Invoke
        prediction = scheduling_tool_calling(
            client=client,
            user_prompt=patient_intention,
            history=chat_history,
        )

        if prediction['type'] == 'tool':
            res = prediction['result']
            st = res['status']
            idx = res['index']

            # Schedule not found case: -> text
            if st is None and idx['pred'] == -1:
                prediction['type'] = 'text'
                prediction['result'] = "Sorry, we couldn't find a matching appointment. Could you please check your appointment details again?"
                return prediction

            # Build retrieval result_dict
            if st is None:  # No GT, retrieved
                result_dict['gt'].append({'reschedule': None})
                result_dict['pred'].append({'reschedule': idx['pred']})
                result_dict['status'].append(None)
                result_dict['status_code'].append(None)
            elif st is False:  # GT exists, identification failed
                result_dict['gt'].append({'reschedule': idx['gt']})
                result_dict['pred'].append({'reschedule': idx['pred']})
                result_dict['status'].append(False)
                result_dict['status_code'].append(STATUS_CODES['reschedule']['identify'])
                prediction['result_dict'] = result_dict
                prediction['tmp_flag'] = 'retrieve'
                return prediction
            else:  # True
                result_dict['gt'].append({'reschedule': idx['gt']})
                result_dict['pred'].append({'reschedule': idx['pred']})
                result_dict['status'].append(True)
                result_dict['status_code'].append(STATUS_CODES['correct'])

            # Translate pipeline action into tmp_flag + result_dict updates
            action = res.get('action')
            if action == 'reschedule':
                result_dict['pred'] = [res['new_schedule']]
                prediction['tmp_flag'] = 'reschedule'
            elif action == 'waiting_list':
                prediction['tmp_flag'] = 'waiting_list'
            elif action == 'schedule_fail':
                result_dict['pred'] = [res['new_schedule']]
                result_dict['status'] = [False]
                result_dict['status_code'] = [STATUS_CODES['reschedule']['schedule'].format(
                    status_code=res.get('schedule_status_code') or STATUS_CODES['format'])]
                prediction['tmp_flag'] = 'schedule'

            prediction['result_dict'] = result_dict
            return prediction

        # Clarification message case -> return: str
        elif prediction['type'] == 'text':
            return prediction

        # Error
        else:
            raise TypeError(colorstr("red", "Error: Unexpected return type from rescheduling method."))
    

    def scheduling_simulate(self,
                            gt_data: dict,
                            staff_known_data: dict,
                            doctor_information: Optional[dict] = None,
                            verbose: bool = False,
                            max_inferences: int = 5,
                            intake_executed: bool = False,
                            natural_express: bool = True,
                            reasoning_max_tries: int = 0,
                            patient_kwargs: dict = {},
                            staff_kwargs: dict = {},
                            **kwargs) -> Tuple[dict, dict, dict]:
        """
        Simulate a multi-turn outpatient scheduling dialogue between a patient agent and an administrative staff agent.

        Args:
            gt_data (dict): Ground-truth patient condition(s) for each dialogue turn.
            staff_known_data (dict): Patient information known to the staff agent.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details. Defaults to None.
            verbose (bool, optional): Whether to log detailed simulation outputs. Defaults to False.
            max_inferences (int, optional): Maximum number of dialogue turns.
            intake_executed (bool, optional): Whether intake ran before this scheduling step (end-to-end). Defaults to False.
            natural_express: (bool, optional): Whether express new schedule as natural or not. Defaults to True.
            reasoning_max_tries (int, optional): Reasoning fallback maximum number of retries. Defaults to 0.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff scheduling function.
            **kwargs: Shared keyword arguments passed to both agents.

        Returns:
            Tuple[dict, dict, dict]: 
                - Doctor information.
                - Result dictionary after scheduling a new appointment.
                - Token statistics.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        # Initialize agents and result dictionary
        if not intake_executed:
            self._init_agents(verbose=verbose)
            self.admin_staff_mas.root.next_step = 'first_visit_scheduling' # Assuming that the intake task was done successfully
        else:
            assert self.admin_staff_mas.root.next_step == 'first_visit_scheduling', \
                colorstr("red", f"Orchestrator `next_step` must be a `first_visit_scheduling`")
        staff_token_callback = TokenUsageCallback()
        self._init_history()
        staff_token_stats = {}
        filtered_doctor_information = self.environment.get_doctor_schedule(
            doctor_information=doctor_information,
            department=staff_known_data['department'],
            fhir_integration=self.fhir_integration and doctor_information is None,
        )
        tool_calling_agent = self.scheduling_agent.build_agent(
            rule=self.rules, 
            doctor_info=filtered_doctor_information
        )
        merged_patient_kwargs = {**patient_kwargs, **kwargs}
        merged_staff_kwargs = {**staff_kwargs, **kwargs}

        # Staff turn closure: captures all of `scheduling`'s simulation-side arguments
        holder = {}
        def staff_turn(user_prompt: str) -> Tuple[str, bool]:
            staff_known_data.update({'patient_intention': user_prompt})
            staff_response = self.scheduling(
                tool_calling_agent,
                staff_known_data,
                doctor_information,
                chat_history=self._to_lc_history('scheduling'),
                reasoning_max_tries=reasoning_max_tries,
                callback=staff_token_callback,
                **merged_staff_kwargs,
            )
            holder['response'] = staff_response
            return self._render_staff_reply(staff_response, 'scheduling', natural_express), False

        # Start conversation
        staff_greet = self.scheduling_agent.appn_greet
        self.dialog_history['scheduling'].append({"role": "Staff", "content": staff_greet})
        self.admin_staff_mas.state.messages.append({"role": "Staff", "content": staff_greet})
        log(f"{staff_role(self.admin_staff_mas.state):<25}: {staff_greet}")

        # Iterate over multiple preferences if exists
        preference_reject_prob = 0.0 if len(gt_data) <= 1 else self.preference_rejection_prob
        try:
            for i, gt_patient_condition in enumerate(gt_data):
                # For the rejection scenario
                if i != 0:
                    self.update_patient_system_prompt(
                        patient_condition=gt_patient_condition,
                        rejected_preference=gt_data[i-1]['preference']
                    )

                tries = 0
                while 1:
                    # Obtain response from patient
                    patient_response = self.patient_agent(
                        self.dialog_history['scheduling'][-1]["content"],
                        using_multi_turn=True,
                        verbose=False,
                        **merged_patient_kwargs,
                    )
                    patient_token_stats = self.patient_agent.client.token_usages
                    self.dialog_history['scheduling'].append({"role": "Patient", "content": patient_response})
                    role = f"{colorstr('green', 'Patient')} ({gt_patient_condition['preference']})"
                    log(f"{role:<25}: {patient_response}")

                    # Scheduling from staff
                    rendered_response, _ = self.admin_staff_mas.chat(
                        user_prompt=patient_response,
                        callback=staff_turn,
                        using_multi_turn=False,
                        verbose=False,
                    )
                    staff_response = holder.pop('response')
                    self.dialog_history['scheduling'].append({"role": "Staff", "content": rendered_response})
                    log(f"{staff_role(self.admin_staff_mas.state):<25}: {rendered_response}")

                    # Token accounting
                    staff_token_stats = self._accumulate_staff_tokens(
                        staff_response, staff_token_stats, staff_token_callback
                    )

                    # A schedule proposal ends this negotiation turn; a clarification keeps it going.
                    if staff_response['type'] == 'tool':
                        pred_schedule = staff_response['result']
                        break

                    tries += 1
                    if tries > max_inferences:
                        result_dict = {
                            'gt': [gt_patient_condition],
                            'pred': [None],
                            'status': [False],
                            'status_code': [STATUS_CODES['simulation']],
                            'dialog': [preprocess_dialog(self.dialog_history['scheduling'])]
                        }
                        token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
                        self._finish_scheduling_turn('scheduling', verbose)
                        return doctor_information, result_dict, token_usage

                # Sanity check
                ## No GT case
                if self.sanity_checker is None:
                    status, status_code = True, STATUS_CODES['correct']
                ## GT existing case
                else:
                    status, status_code = self.sanity_checker.schedule_check(
                        prediction=pred_schedule,
                        gt_patient_condition=gt_patient_condition,
                        doctor_information=doctor_information,
                        environment=self.environment
                    )

                if not status:
                    break

                # Preference rejection logic
                ## Rejection case
                _schedule = pred_schedule['schedule']
                _doctor = list(_schedule.keys())[0]
                _date = _schedule[_doctor]['date']
                if random.random() < preference_reject_prob and i != len(gt_data) - 1 and \
                    gt_data[i+1]['preferred_doctor'] != _doctor and gt_data[i+1]['valid_from'] != _date:  # Avoid overlap with next preferred doctor or next preferred date
                    preference_reject_prob *= self.preference_rejection_prob_decay
                ## Non-rejection case
                else:
                    if natural_express:
                        self.update_patient_system_prompt(
                            new_system_prompt=self.patient_satisfaction_system_prompt
                        )
                        patient_response = self.patient_agent(
                            self.natural_end_phrase.format(schedule=self.dialog_history['scheduling'][-1]['content']),
                            using_multi_turn=True,
                            verbose=False,
                            **merged_patient_kwargs,
                        )
                        patient_token_stats = self.patient_agent.client.token_usages

                    else:
                        patient_response = self.end_phrase

                    self.dialog_history['scheduling'].append({"role": "Patient", "content": patient_response})
                    role = f"{colorstr('green', 'Patient')} ({gt_data[i]['preference']})"
                    log(f"{role:<25}: {patient_response}")

                    break

        except SchedulingError:
            result_dict = {
                'gt': [gt_patient_condition],
                'pred': [None],
                'status': [False],
                'status_code': [STATUS_CODES['format']],
                'dialog': [preprocess_dialog(self.dialog_history['scheduling'])]
            }
            log("Simulation completed.", color=True)
            token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
            self._finish_scheduling_turn('scheduling', verbose)
            return doctor_information, result_dict, token_usage

        # Otherwise
        except Exception as e:
            status_code = STATUS_CODES['unexpected'].format(e=e)
            log(status_code, level='warning')
            result_dict = {
                'gt': [gt_patient_condition],
                'pred': [None],
                'status': [False],
                'status_code': [status_code],
                'dialog': [preprocess_dialog(self.dialog_history['scheduling'])]
            }

        # Oranize the result
        ## Defaults to failure case dictionary
        result_dict = {
            'gt': [gt_patient_condition],
            'pred': [pred_schedule],
            'status': [False],
            'status_code': [status_code],
            'dialog': [preprocess_dialog(self.dialog_history['scheduling'])]
        }
        ## Success case
        if status:
            try:
                pred_doctor_name = list(pred_schedule['schedule'].keys())[0]
                schedule = pred_schedule['schedule'][pred_doctor_name]
                prediction = {
                    'visit_type': 'first_visit',
                    'patient': staff_known_data['patient'],
                    'attending_physician': pred_doctor_name,
                    'department': staff_known_data['department'],
                    'date': schedule['date'],
                    'schedule': [schedule['start'], schedule['end']],
                    'patient_intention': staff_known_data['patient_intention'],
                    'preference': gt_data[i].get('preference'),
                    'preferred_doctor': gt_data[i].get('preferred_doctor'),
                    'valid_from': gt_data[i].get('valid_from'),
                    'test': None,
                    'last_updated_time': self.environment.current_time
                }
                result_dict['pred'] = [prediction]
                result_dict['status'] = [True]
            except:
                result_dict['status_code'] = [STATUS_CODES['format']]
                log('No sanity checker is available; an error occurred while parsing the prediction. Returning a failure result.', level='warning')

        log("Simulation completed.", color=True)
        token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
        self._finish_scheduling_turn('scheduling', verbose)
        return doctor_information, result_dict, token_usage

    
    def canceling_simulate(self, 
                           gt_idx: Optional[int] = None,
                           doctor_information: Optional[dict] = None,
                           patient_schedules: Optional[list[dict]] = None,
                           verbose: bool = True,
                           max_inferences: int = 5,
                           patient_kwargs: dict = {},
                           staff_kwargs: dict = {},
                           **kwargs) -> Tuple[dict, dict]:
        """
        Simulate a multi-turn conversation for appointment cancellation.

        Args:
            gt_idx (Optional[int], optional): Ground-truth index of the appointment to be cancelled. Defaults to None.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s).
            patient_schedules (Optional[list[dict]], optional): List of patient appointment schedules. Defaults to None.
            verbose (bool, optional): Whether to print conversation logs. Defaults to True.
            max_inferences (int, optional): Maximum number of dialogue turns.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff agent.
            **kwargs: Additional keyword arguments passed to the patient and staff agent.

        Raises:
            DataNotFoundError: Schedule not found error.

        Returns:
            Tuple[dict, dict]: Updated doctor information and a result dictionary after cancellation.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        # Initialize agents and result dictionary
        result_dict = init_result_dict()
        self._init_history()
        self._init_agents(verbose=verbose)
        patient_schedules = self.environment.patient_schedules if patient_schedules is None else patient_schedules
        doctor_information = self.environment.get_general_doctor_info_from_fhir() if self.fhir_integration else doctor_information
        tool_calling_agent = self.scheduling_agent.build_agent(
            rule=self.rules, 
            doctor_info=doctor_information,
            patient_schedule_list=patient_schedules,
            gt_idx=gt_idx,
        )
        merged_patient_kwargs = {**patient_kwargs, **kwargs}

        # Staff turn closure: captures all of `scheduling`'s simulation-side arguments
        holder = {}
        def staff_turn(user_prompt: str) -> Tuple[str, bool]:
            staff_response = self.canceling(
                client=tool_calling_agent,
                patient_intention=user_prompt,
                chat_history=self._to_lc_history('cancel')
            )
            holder['response'] = staff_response
            return self._render_staff_reply(staff_response, 'cancel'), False

        # Start conversation
        staff_greet = self.admin_staff_mas.root.agent.staff_greet
        self.dialog_history['cancel'].append({"role": "Staff", "content": staff_greet})
        self.admin_staff_mas.state.messages.append({"role": "Staff", "content": staff_greet})
        log(f"{staff_role(self.admin_staff_mas.state):<25}: {staff_greet}")

        try:
            for _ in range(max_inferences):
                # Obtain response from patient
                patient_response = self.patient_agent(
                    self.dialog_history['cancel'][-1]["content"],
                    using_multi_turn=True,
                    verbose=False,
                    **merged_patient_kwargs
                )
                self.dialog_history['cancel'].append({"role": "Patient", "content": patient_response})
                role = f"{colorstr('green', 'Patient')} (cancel)"
                log(f"{role:<25}: {patient_response}")

                # Canceling from staff
                rendered_response, _ = self.admin_staff_mas.chat(
                    user_prompt=patient_response,
                    callback=staff_turn,
                    using_multi_turn=False,
                    verbose=False,
                )
                staff_response = holder.pop('response')

                # A wrong identification cancels nothing -> surface as a not-found failure
                if staff_response['type'] == 'tool' and staff_response['result_dict']['status'][0] is False:
                    raise DataNotFoundError(colorstr("red", "Error: Schedule not found error."))

                self.dialog_history['cancel'].append({"role": "Staff", "content": rendered_response})
                log(f"{staff_role(self.admin_staff_mas.state):<25}: {rendered_response}")

                # Tool calling result -> successful cancellation (a clarification 'text' reply just re-iterates)
                if staff_response['type'] == 'tool':
                    result_dict = staff_response['result_dict']

                    # Final response of patient
                    self.dialog_history['cancel'].append({"role": "Patient", "content": self.end_phrase})
                    role = f"{colorstr('green', 'Patient')} (cancel)"
                    log(f"{role:<25}: {self.end_phrase}")

                    result_dict['dialog'].append(preprocess_dialog(self.dialog_history['cancel']))
                    break
                
            # The case without any determination during the simulation
            if not len(result_dict['gt']):
                result_dict = {
                    'gt': [{'cancel': gt_idx}],
                    'pred': [None],
                    'status': [False],
                    'status_code': [STATUS_CODES['cancel']['identify']],
                    'dialog': [preprocess_dialog(self.dialog_history['cancel'])]
                }
                
        # Requested schedule indentification error
        except DataNotFoundError:
            result_dict['dialog'].append(preprocess_dialog(self.dialog_history['cancel']))

        # Tool calling error
        except TypeError:
            result_dict = {
                'gt': [{'cancel': gt_idx}],
                'pred': [None],
                'status': [False],
                'status_code': [STATUS_CODES['cancel']['type']],
                'dialog': [preprocess_dialog(self.dialog_history['cancel'])]
            }

        # Otherwhise
        except Exception as e:
            status_code = STATUS_CODES['unexpected'].format(e=e)
            log(status_code, level='warning')
            result_dict = {
                'gt': [{'cancel': gt_idx}],
                'pred': [None],
                'status': [False],
                'status_code': [status_code],
                'dialog': [preprocess_dialog(self.dialog_history['cancel'])]
            }

        log("Simulation completed.", color=True)
        self._finish_scheduling_turn('cancel', verbose)
        return doctor_information, result_dict


    def rescheduling_simulate(self, 
                              gt_idx: Optional[int] = None,
                              doctor_information: Optional[dict] = None,
                              patient_schedules: Optional[list[dict]] = None,
                              verbose: bool = True,
                              max_inferences: int = 5,
                              patient_kwargs: dict = {},
                              staff_kwargs: dict = {},
                              **kwargs) -> Tuple[dict, dict]:
        """
        Simulate a multi-turn conversation for appointment rescheduling.

        Args:
            gt_idx (Optional[int], optional): Ground-truth index of the appointment to be rescheduled. Defaults to None.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s).
            patient_schedules (Optional[list[dict]], optional): List of patient appointment schedules. Defaults to None.
            verbose (bool, optional): Whether to print conversation logs. Defaults to True.
            max_inferences (int, optional): Maximum number of dialogue turns.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff agent.
            **kwargs: Additional keyword arguments passed to the patient and staff agents.

        Raises:
            TypeError: If the return type from the rescheduling method is unexpected.
            DataNotFoundError: Schedule not found error.
            SchedulingError: Scheduling error.
            
        Returns:
            Tuple[dict, dict]: Updated doctor information and a result dictionary after cancellation.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        # Initialize agents and result dictionary
        result_dict = init_result_dict()
        self._init_history()
        self._init_agents(verbose=verbose)
        patient_schedules = self.environment.patient_schedules if patient_schedules is None else patient_schedules
        doctor_information = self.environment.get_general_doctor_info_from_fhir() if self.fhir_integration else doctor_information
        merged_patient_kwargs = {**patient_kwargs, **kwargs}
        merged_staff_kwargs = {**staff_kwargs, **kwargs}
        tool_calling_agent = self.scheduling_agent.build_agent(
            rule=self.rules,
            doctor_info=doctor_information,
            patient_schedule_list=patient_schedules,
            gt_idx=gt_idx,
            reschedule_pipeline=self._make_reschedule_pipeline(doctor_information, **merged_staff_kwargs),
        )

        # Staff turn closure: captures all of `scheduling`'s simulation-side arguments
        holder = {}
        def staff_turn(user_prompt: str) -> Tuple[str, bool]:
            staff_response = self.rescheduling(
                client=tool_calling_agent,
                patient_intention=user_prompt,
                doctor_information=doctor_information,
                chat_history=self._to_lc_history('reschedule'),
                **merged_staff_kwargs,
            )
            holder['response'] = staff_response
            return self._render_staff_reply(staff_response, 'reschedule'), False

        # Start conversation
        staff_greet = self.admin_staff_mas.root.agent.staff_greet
        self.dialog_history['reschedule'].append({"role": "Staff", "content": staff_greet})
        self.admin_staff_mas.state.messages.append({"role": "Staff", "content": staff_greet})
        log(f"{staff_role(self.admin_staff_mas.state):<25}: {staff_greet}")

        try:
            for _ in range(max_inferences):
                # Obtain response from patient
                patient_response = self.patient_agent(
                    self.dialog_history['reschedule'][-1]["content"],
                    using_multi_turn=True,
                    verbose=False,
                    **merged_patient_kwargs
                )
                self.dialog_history['reschedule'].append({"role": "Patient", "content": patient_response})
                role = f"{colorstr('green', 'Patient')} (move)"
                log(f"{role:<25}: {patient_response}")

                # Rescheduling from staff
                rendered_response, _ = self.admin_staff_mas.chat(
                    user_prompt=patient_response,
                    callback=staff_turn,
                    using_multi_turn=False,
                    verbose=False,
                )
                staff_response = holder.pop('response')

                # Tool-calling failures resolve nothing -> surface them before recording a staff turn.
                if staff_response['type'] == 'tool':
                    tmp_flag = staff_response.get('tmp_flag')
                    if tmp_flag == 'retrieve':
                        result_dict = staff_response['result_dict']
                        raise DataNotFoundError(colorstr("red", "Error: Schedule not found error."))
                    elif tmp_flag == 'schedule':
                        result_dict = staff_response['result_dict']
                        raise SchedulingError(colorstr("red", "Error: Scheduling error."))
                    elif tmp_flag not in ('waiting_list', 'reschedule'):
                        raise TypeError(colorstr("red", "Error: Unexpected return type from rescheduling method."))

                self.dialog_history['reschedule'].append({"role": "Staff", "content": rendered_response})
                log(f"{staff_role(self.admin_staff_mas.state):<25}: {rendered_response}")

                # Tool calling result -> successful reschedule / waiting-list
                if staff_response['type'] == 'tool':
                    result_dict = staff_response['result_dict']

                    # Final response of patient
                    self.dialog_history['reschedule'].append({"role": "Patient", "content": self.end_phrase})
                    role = f"{colorstr('green', 'Patient')} (move)"
                    log(f"{role:<25}: {self.end_phrase}")

                    result_dict['dialog'].append(preprocess_dialog(self.dialog_history['reschedule']))
                    break

            # The case without any determination during the simulation
            if not len(result_dict['gt']):
                result_dict = {
                    'gt': [{'reschedule': gt_idx}],
                    'pred': [None],
                    'status': [False],
                    'status_code': [STATUS_CODES['reschedule']['identify']],
                    'dialog': [preprocess_dialog(self.dialog_history['reschedule'])]
                }

        # Requested schedule indentification error
        except DataNotFoundError:
            result_dict['dialog'].append(preprocess_dialog(self.dialog_history['reschedule']))
        
        # Scheduling Error:
        except SchedulingError:
            result_dict['dialog'].append(preprocess_dialog(self.dialog_history['reschedule']))
        
        # Tool calling error
        except TypeError:
            result_dict = {
                'gt': [{'reschedule': gt_idx}],
                'pred': [None],
                'status': [False],
                'status_code': [STATUS_CODES['reschedule']['type']],
                'dialog': [preprocess_dialog(self.dialog_history['reschedule'])]
            }
        
        # Otherwise
        except Exception as e:
            status_code = STATUS_CODES['unexpected'].format(e=e)
            log(status_code, level='warning')
            result_dict = {
                'gt': [{'reschedule': gt_idx}],
                'pred': [None],
                'status': [False],
                'status_code': [status_code],
                'dialog': [preprocess_dialog(self.dialog_history['reschedule'])]
            }
            
        log("Simulation completed.", color=True)
        self._finish_scheduling_turn('reschedule', verbose)
        return doctor_information, result_dict
    

    def scheduling_simulate_stream(self,
                                   gt_data: dict,
                                   staff_known_data: dict,
                                   doctor_information: Optional[dict] = None,
                                   verbose: bool = False,
                                   max_inferences: int = 5,
                                   intake_executed: bool = True,
                                   natural_express: bool = True,
                                   reasoning_max_tries: int = 3,
                                   patient_kwargs: dict = {},
                                   staff_kwargs: dict = {},
                                   **kwargs):
        """
        Simulate a multi-turn outpatient scheduling dialogue between a patient agent and an administrative staff agent.

        Args:
            gt_data (dict): Ground-truth patient condition(s) for each dialogue turn.
            staff_known_data (dict): Patient information known to the staff agent.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details. Defaults to None.
            verbose (bool, optional): Whether to log detailed simulation outputs. Defaults to False.
            max_inferences (int, optional): Maximum number of dialogue turns.
            intake_executed (bool, optional): Whether intake ran before this scheduling step (end-to-end). Defaults to True.
            natural_express (bool, optional): Whether express new schedule as natural or not. Defaults to True.
            reasoning_max_tries (int, optional): Reasoning fallback maximum number of retries. Defaults to 3.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff scheduling function.
            **kwargs: Shared keyword arguments passed to both agents.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        # Initialize agents and result dictionary
        if not intake_executed:
            self._init_agents(verbose=verbose)
            self.admin_staff_mas.root.next_step = 'first_visit_scheduling' # Assuming that the intake task was done successfully
        else:
            assert self.admin_staff_mas.root.next_step == 'first_visit_scheduling', \
                colorstr("red", f"Orchestrator `next_step` must be a `first_visit_scheduling`")
        staff_token_callback = TokenUsageCallback()
        self._init_history()
        staff_token_stats = {}
        filtered_doctor_information = self.environment.get_doctor_schedule(
            doctor_information=doctor_information,
            department=staff_known_data['department'],
            fhir_integration=self.fhir_integration and doctor_information is None,
        )
        tool_calling_agent = self.scheduling_agent.build_agent(
            rule=self.rules, 
            doctor_info=filtered_doctor_information
        )
        merged_patient_kwargs = {**patient_kwargs, **kwargs}
        merged_staff_kwargs = {**staff_kwargs, **kwargs}
        
        # Start conversation
        staff_greet = self.scheduling_agent.appn_greet
        self.dialog_history['scheduling'].append({"role": "Staff", "content": staff_greet})
        self.admin_staff_mas.state.messages.append({"role": "Staff", "content": staff_greet})
        log(f"{staff_role(self.admin_staff_mas.state):<25}: {staff_greet}")

        # Staff turn closure: routes the structured staff scheduling turn through the MAS.
        holder = {}
        def staff_turn(user_prompt: str) -> Tuple[str, bool]:
            staff_known_data.update({'patient_intention': user_prompt})
            staff_response = run_with_retry(
                self.scheduling,
                tool_calling_agent,
                staff_known_data,
                doctor_information,
                chat_history=self._to_lc_history('scheduling'),
                reasoning_max_tries=reasoning_max_tries,
                callback=staff_token_callback,
                max_retries=5,
                **merged_staff_kwargs,
            )
            holder['response'] = staff_response
            return self._render_staff_reply(staff_response, 'scheduling', natural_express), False

        # Iterate over multiple preferences if exists
        preference_reject_prob = 0.0 if len(gt_data) <= 1 else self.preference_rejection_prob
        for i, gt_patient_condition in enumerate(gt_data):
            # For the rejection scenario
            if i != 0:
                self.update_patient_system_prompt(
                    patient_condition=gt_patient_condition,
                    rejected_preference=gt_data[i-1]['preference']
                )

            tries = 0
            while 1:
                # Obtain response from patient
                patient_response = run_with_retry(
                    self.patient_agent,
                    self.dialog_history['scheduling'][-1]["content"],
                    using_multi_turn=True,
                    verbose=False,
                    max_retries=5,
                    **merged_patient_kwargs,
                )
                self.dialog_history['scheduling'].append({"role": "Patient", "content": patient_response})
                role = f"{colorstr('green', 'Patient')} ({gt_patient_condition['preference']})"
                log(f"{role:<25}: {patient_response}")
                yield 'Patient', preprocess_utterance(patient_response), None
                
                # Scheduling from staff (routed through the MAS orchestrator)
                rendered_response, _ = self.admin_staff_mas.chat(
                    user_prompt=patient_response,
                    callback=staff_turn,
                    using_multi_turn=False,
                    verbose=False,
                )
                staff_response = holder.pop('response')
                staff_token_stats = self._accumulate_staff_tokens(
                    staff_response, staff_token_stats, staff_token_callback
                )

                # Record the staff utterance rendered by `_render_staff_reply`
                self.dialog_history['scheduling'].append({"role": "Staff", "content": rendered_response})
                log(f"{staff_role(self.admin_staff_mas.state):<25}: {rendered_response}")

                # A schedule proposal ends this negotiation turn; a clarification keeps it going.
                if staff_response['type'] == 'tool':
                    pred_schedule = staff_response['result']
                    yield 'Staff', preprocess_utterance(rendered_response), pred_schedule
                    break

                yield 'Staff', preprocess_utterance(rendered_response), None
                tries += 1
                if tries > max_inferences:
                    self._finish_scheduling_turn('scheduling', verbose)
                    return
            
            # Preference rejection logic
            ## Rejection case
            _schedule = pred_schedule['schedule']
            _doctor = list(_schedule.keys())[0]
            _date = _schedule[_doctor]['date']
            if random.random() < preference_reject_prob and i != len(gt_data) - 1 and \
                gt_data[i+1]['preferred_doctor'] != _doctor and gt_data[i+1]['valid_from'] != _date:  # Avoid overlap with next preferred doctor or next preferred date
                preference_reject_prob *= self.preference_rejection_prob_decay
            
            ## Non-rejection case
            else:
                if natural_express:
                    final_preference = gt_patient_condition.get('preference')
                    final_preferred_condition = gt_patient_condition.get('valid_from') if final_preference == 'date' \
                        else gt_patient_condition.get('preferred_doctor')

                    if final_preference.lower() == 'asap':
                        self.update_patient_system_prompt(
                            new_system_prompt=self.patient_satisfaction_system_prompt
                        )
                        patient_response = run_with_retry(
                            self.patient_agent,
                            self.natural_end_phrase.format(schedule=self.dialog_history['scheduling'][-1]['content']),
                            using_multi_turn=True,
                            verbose=False,
                            max_retries=5,
                            **merged_patient_kwargs,
                        )
                        self.dialog_history['scheduling'].append({"role": "Patient", "content": patient_response})
                        role = f"{colorstr('green', 'Patient')} ({gt_data[i]['preference']})"
                        log(f"{role:<25}: {patient_response}")
                        yield 'Patient', preprocess_utterance(patient_response), None

                    else:
                        # doctor or date preference - evaluate with retry
                        accept_tries = 0
                        while accept_tries <= max_inferences:
                            self.update_patient_system_prompt(
                                new_system_prompt=self.patient_evaluation_system_prompt.format(
                                    preference=final_preference,
                                    preferred_condition=final_preferred_condition
                                )
                            )
                            eval_phrase = self.patient_schedule_evaluation_phrase.format(
                                schedule=self.dialog_history['scheduling'][-1]['content'],
                                preference=final_preference,
                                preferred_condition=final_preferred_condition
                            )

                            patient_response = run_with_retry(
                                self.patient_agent,
                                eval_phrase,
                                using_multi_turn=True,
                                verbose=False,
                                max_retries=5,
                                **merged_patient_kwargs,
                            )

                            self.dialog_history['scheduling'].append({"role": "Patient", "content": patient_response})
                            role = f"{colorstr('green', 'Patient')} ({gt_data[i]['preference']})"
                            log(f"{role:<25}: {patient_response}")
                            yield 'Patient', preprocess_utterance(patient_response), None

                            if '#ACCEPT' in patient_response:
                                break

                            # Not accepted -> staff reschedules (routed through the MAS orchestrator)
                            rendered_response, _ = self.admin_staff_mas.chat(
                                user_prompt=patient_response,
                                callback=staff_turn,
                                using_multi_turn=False,
                                verbose=False,
                            )
                            staff_response = holder.pop('response')
                            staff_token_stats = self._accumulate_staff_tokens(
                                staff_response, staff_token_stats, staff_token_callback
                            )
                            self.dialog_history['scheduling'].append({"role": "Staff", "content": rendered_response})
                            log(f"{staff_role(self.admin_staff_mas.state):<25}: {rendered_response}")

                            if staff_response['type'] == 'tool':
                                pred_schedule = staff_response['result']
                                yield 'Staff', preprocess_utterance(rendered_response), pred_schedule
                            else:
                                yield 'Staff', preprocess_utterance(rendered_response), None

                            accept_tries += 1

                else:
                    patient_response = self.end_phrase
                    self.dialog_history['scheduling'].append({"role": "Patient", "content": patient_response})
                    role = f"{colorstr('green', 'Patient')} ({gt_data[i]['preference']})"
                    log(f"{role:<25}: {patient_response}")
                    yield 'Patient', preprocess_utterance(patient_response), None

                break

        self._finish_scheduling_turn('scheduling', verbose)
