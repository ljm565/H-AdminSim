import re
import os
import json
import random
from importlib import resources
from patientsim import PatientAgent
from decimal import Decimal, getcontext
from typing import Tuple, Union, Optional
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

from h_adminsim import AdminStaffAgent
from h_adminsim.registry.errors import ToolCallingError, ScheduleNotFoundError
from h_adminsim.registry import PREFERENCE_PHRASE_PATIENT, PREFERENCE_PHRASE_STAFF, STATUS_CODES
from h_adminsim.environment.hospital import HospitalEnvironment
from h_adminsim.utils import log, colorstr
from h_adminsim.tools.sanity_checker import SanityChecker
from h_adminsim.tools import SchedulingRule, scheduling_tool_calling
from h_adminsim.utils.common_utils import *



class OPScehdulingSimulation:
    def __init__(self,
                 patient_agent: PatientAgent,
                 admin_staff_agent: AdminStaffAgent,
                 metadata: dict,
                 department_data: dict,
                 environment: HospitalEnvironment,
                 preference_rejection_prob: float = 0.3,
                 preferene_rejection_prob_decay: float = 0.5,
                 fhir_integration: bool = False,
                 schedule_rejection_prompt_path: Optional[str] = None,
                 sanity_checker: Optional[SanityChecker] = None):
        
        # Initialize simulation parameters
        getcontext().prec = 10
        self.patient_agent = patient_agent
        self.admin_staff_agent = admin_staff_agent
        self.environment = environment
        self._START_HOUR = metadata['time']['start_hour']
        self._END_HOUR = metadata['time']['end_hour']
        self._TIME_UNIT = metadata['time']['interval_hour']
        self._DAY = metadata['days']
        self.preference_rejection_prob = preference_rejection_prob
        self.preferene_rejection_prob_decay = preferene_rejection_prob_decay
        self.fhir_integration = fhir_integration
        self.rejection_system_prompt_template = self._init_prompt(schedule_rejection_prompt_path)
        self.sanity_checker = sanity_checker
        self.rules = SchedulingRule(metadata, department_data, self.environment, self.fhir_integration)
        self.end_phrase = "Thank you."
        self._init_history()

    
    def _init_prompt(self, schedule_rejection_prompt_path: Optional[str] = None) -> str:
        """
        Initialize the schedule rejection system prompt for the administration staff agent.

        Args:
            schedule_rejection_prompt_path (Optional[str], optional): Path to a custom schedule rejection system prompt file. 
                                                                      If not provided, the default system prompt will be used. Defaults to None.

        Returns:
            str: Schedule rejection prompt template.
        
        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.
        """
        # Initialilze with the default system prompt
        if not schedule_rejection_prompt_path:
            prompt_file_name = "schedule_patient_rejected_system.txt"
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            rejection_system_prompt_template = file_path.read_text()
        
        # User can specify a custom system prompt
        else:
            if not os.path.exists(schedule_rejection_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {schedule_rejection_prompt_path}"))
            with open(schedule_rejection_prompt_path, 'r') as f:
                rejection_system_prompt_template = f.read()
        return rejection_system_prompt_template


    def _init_agents(self, verbose: bool = True):
        """
        Reset the conversation histories and token usage records of both the Patient and Doctor agents.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        """
        self.patient_agent.reset_history(verbose=verbose)
        self.admin_staff_agent.reset_history(verbose=verbose)


    def _init_history(self):
        """
        Reset the dialogue histories.
        """
        self.dialog_history = {
            'scheduling': [],
            'cancel': [],
            'reschedule': [],
        }

    
    def _update_result_dict(self, data: dict):
        """
        Update the result dictionary with the provided data.

        Args:
            data (dict): A dictionary containing the data to update the result dictionary with.
        """
        for k in data.keys():
            self.result_dict[k] = data[k]

    
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
    

    def update_from_kwargs(self, **kwargs):
        """
        Update simulation parameters from keyword arguments.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    
    def postprocessing(self, 
                       strategy: str, 
                       data: Union[str, dict],
                       filtered_doctor_information: Optional[dict] = None) -> Union[str, dict]:
        """
        Attempts to parse the given text as JSON. If parsing succeeds, returns a dictionary;
        otherwise, returns the original string.

        Args:
            strategy (str): Scheduling strategy. It must be either `llm` or `tool_calling`.
            data (Union[str, dict]): The text output to post-process, potentially a JSON-formatted string. 
            filtered_doctor_information (Optional[dict], optional): Department-filtered doctor information 
                                                                    to postprocess the schedule by tool_calling strategy.

        Returns:
            Union[str, dict]: A dictionary if the text is valid JSON, otherwise the original string.
        """
        if strategy == 'llm':
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
                                     patient_condition: dict,
                                     rejected_preference: str):
        """
        Update a system prompt of the patient agent for proposed schedule rejection scenario.

        Args:
            patient_condition (dict): Patient ground-truth condition including current preference.
            rejected_preference (str): The scheduling preference proposed by the staff agent in the previous turn
                                       that the patient must explicitly reject.
        """
        # Build new system prompts for rejection scenario
        preference = patient_condition.get('preference')
        preference_desc = PREFERENCE_PHRASE_PATIENT[preference] if preference != 'date' \
                else PREFERENCE_PHRASE_PATIENT[preference].format(date=patient_condition.get('valid_from'))
        rejected_preference_desc = PREFERENCE_PHRASE_STAFF[rejected_preference] if rejected_preference != 'date' \
                else PREFERENCE_PHRASE_STAFF[rejected_preference].format(date='a specific date')    
        system_prompt = self.rejection_system_prompt_template.format(
            preference=preference,
            preference_desc=preference_desc,
            preferred_doctor=patient_condition['preferred_doctor'],
            rejected_preference=rejected_preference_desc,
            personality=self.patient_agent.personality,
        )

        # Update new system prompts for rejection scenario
        self.patient_agent.system_prompt = system_prompt
        if len(self.patient_agent.client.histories) and \
            isinstance(self.patient_agent.client.histories[0], dict) and \
                self.patient_agent.client.histories[0].get('role') == 'system':
            self.patient_agent.client.histories[0]['content'][0]['text'] = system_prompt


    def scheduling(self,
                   client: AgentExecutor,
                   known_condition: dict,
                   doctor_information: Optional[dict] = None, 
                   reschedule_flag: bool = False,
                   chat_history: list = [],
                   **kwargs) -> Union[str, dict]:
        """
        Make an appointment between the doctor and the patient.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            known_condition (dict): Patient conditions known to the staff.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details. Defaults to None.
            reschedule_flag (bool, optional): Whether this process is rescheduling or not. Defaults to False.
            chat_history (list, optional): Chat history. Defaults to [].

        Return
            Union[str, dict]: The original prediction (wrong case) or processed prediction (correct case) (either unchanged or used for debugging/logging if invalid).
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, log(f"Doctor information must be provided if you don't use FHIR.", level="error")

        # Initialization based on the known condition from the staff
        department = known_condition['department']
        filtered_doctor_information = self.environment.get_doctor_schedule(
            doctor_information=doctor_information if not self.fhir_integration else None,
            department=department,
            fhir_integration=self.fhir_integration,
        )
        
        # First, try to use the tool calling
        try:
            # Invoke
            prediction = scheduling_tool_calling(
                client=client, 
                user_prompt=known_condition['patient_intention'],
                history=chat_history
            )

            # Post-processing
            ## Scheduling result
            if prediction['type'] == 'tool':
                schedule = self.postprocessing(
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
                raise TypeError(colorstr("red", "Error: Unexpected return type from canceling method."))

        # If tool calling fails, fallback to LLM-based scheduling
        except ToolCallingError:
            log('Failed to select an appropriate tool. Falling back to reasoning-based scheduling.', level='warning')
            reschedule_desc = "Rescheduling requested. This is the rescheduling of a patient who wishes to move their appointment earlier due to a previous patient's cancelled reservation" \
                if reschedule_flag else 'Not requested.'
            filtered_doctor_information = self.environment.get_doctor_schedule(
                doctor_information=doctor_information if not self.fhir_integration else None,
                department=department,
                fhir_integration=self.fhir_integration,
                express_detail=True
            )
            user_prompt = self.admin_staff_agent.scheduling_user_prompt_template.format(
                START_HOUR=self._START_HOUR,
                END_HOUR=self._END_HOUR,
                TIME_UNIT=self._TIME_UNIT,
                CURRENT_TIME=self.environment.current_time,
                DEPARTMENT=department,
                PREFERENCE=known_condition['patient_intention'],
                RESCHEDULING_FLAG=reschedule_desc,
                DAY=self._DAY,
                DOCTOR=json.dumps(filtered_doctor_information, indent=2),
            )
            schedule = self.admin_staff_agent(
                user_prompt,
                using_multi_turn=False,
                verbose=False,
                **kwargs,
            )
            schedule = self.postprocessing(
                strategy='llm',
                data=schedule,
            )
            prediction['result'] = schedule
            self.admin_staff_agent.reset_history(verbose=False)

        return prediction
    

    def canceling(self, 
                  client: AgentExecutor,
                  patient_intention: str,
                  chat_history: list = []) -> Union[dict, str]:
        """
        Handle a multi-turn appointment cancellation request using a tool-calling agent.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            patient_intention (str): The patient's utterance expressing a rescheduling request.
            chat_history (list, optional): Chat history. Defaults to [].

        Returns:
            Union[dict, str]: The cancellation result or a clarification message to the patient.

        Raises:
            TypeError: If the returned type is not supported.
        """
        # Invoke
        prediction = scheduling_tool_calling(
            client=client,
            user_prompt=patient_intention,
            history=chat_history,
        )
        
        # Canceling result
        if prediction['type'] == 'tool':
            # Schedule not found case: -> return: str
            if prediction['result']['result_dict']['pred'][0]['cancel'] == -1:
                prediction['type'] = 'text'
                prediction['result'] = "Sorry, we couldn't find a matching appointment. Could you please check your appointment details again?"
                return prediction
            
            # Successful cancellation case -> return: dict
            else:
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
                     chat_history: list = [],
                     doctor_information: Optional[dict] = None,
                     **kwargs) -> Union[dict, str]:
        """
        Handle a multi-turn appointment rescheduling request using a tool-calling agent.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            patient_intention (str): The patient's utterance expressing a rescheduling request.
            chat_history (list, optional): Chat history. Defaults to [].
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details. Defaults to None.

        Returns:
            Union[dict, str]: The rescheduling result or a clarification message to the patient.
        
        Raises:
            TypeError: If the returned type is not supported.
        """
        # Invoke
        prediction = scheduling_tool_calling(
            client=client,
            user_prompt=patient_intention,
            history=chat_history,
        )

        # Rescheduling result
        if prediction['type'] == 'tool':
            # Schedule not found case: -> return: str
            if prediction['result']['result_dict']['pred'][0]['reschedule'] == -1:
                prediction['type'] = 'text'
                prediction['result'] = "Sorry, we couldn't find a matching appointment. Could you please check your appointment details again?"
                return prediction
        
            # Successfully retrieve the original schedule -> return: dict
            else:
                # Step 1: Retrieved original schedule
                original_schedule = prediction['result']['original_schedule']

                # Step 2: Try to reschedule based on the patient record (or the memo)
                filtered_doctor_information = self.environment.get_doctor_schedule(
                    doctor_information=doctor_information if not self.fhir_integration else None,
                    department=original_schedule['department'],
                    fhir_integration=self.fhir_integration,
                )
                _schedule_client = self.admin_staff_agent.build_agent(
                    rule=self.rules, 
                    doctor_info=filtered_doctor_information,
                    only_schedule_tool=True
                )
                new_schedule = self.scheduling(
                    client=_schedule_client,
                    known_condition=original_schedule,
                    doctor_information=doctor_information,
                    reschedule_flag=True,
                    **kwargs
                )
                prediction['result']['new_schedule'] = new_schedule['result']
                return prediction

        # Clarification message case -> return: str
        elif prediction['type'] == 'text':
            return prediction

        # Error
        else:
            raise TypeError(colorstr("red", "Error: Unexpected return type from canceling method."))
    

    def scheduling_simulate(self,
                            gt_data: dict,
                            staff_known_data: dict,
                            doctor_information: Optional[dict] = None,
                            verbose: bool = False,
                            patient_kwargs: dict = {},
                            staff_kwargs: dict = {},
                            **kwargs):
        """
        Simulate a multi-turn outpatient scheduling dialogue between a patient agent and an administrative staff agent.

        Args:
            gt_data (dict): Ground-truth patient condition(s) for each dialogue turn.
            staff_known_data (dict): Patient information known to the staff agent at each turn.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details. Defaults to None.
            verbose (bool, optional): Whether to log detailed simulation outputs. Defaults to False.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff scheduling function.
            **kwargs: Shared keyword arguments passed to both agents.

        Yields:
            dict: Scheduling proposal generated by the staff agent at each turn.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, log(f"Doctor information must be provided if you don't use FHIR.", level="error")

        # Initialize agents and result dictionary
        self._init_agents(verbose=verbose)
        filtered_doctor_information = self.environment.get_doctor_schedule(
            doctor_information=doctor_information if not self.fhir_integration else None,
            department=staff_known_data[0]['department'],
            fhir_integration=self.fhir_integration,
        )
        client = self.admin_staff_agent.build_agent(
            rule=self.rules, 
            doctor_info=filtered_doctor_information
        )
        
        # Sanity check for the simulation
        assert len(gt_data) == len(staff_known_data), \
            log(f"The lengths of gt_data and staff_known_data must be the same, but got gt_data length: {len(gt_data)} and staff_known_data length: {len(staff_known_data)}", level="error")

        # Start conversation
        staff_greet = self.admin_staff_agent.staff_greet
        self.dialog_history['scheduling'].append({"role": "Staff", "content": staff_greet})
        role = f"{colorstr('blue', 'Staff')}"
        log(f"{role:<25}: {staff_greet}")

        # Iterate over multiple preferences if exists
        preference_reject_prob = 0.0 if len(gt_data) <= 1 else self.preference_rejection_prob
        for i, (gt_patient_condition, staff_known_condition) in enumerate(zip(gt_data, staff_known_data)):
            # For the rejection scenario
            if i != 0:
                self.update_patient_system_prompt(
                    patient_condition=gt_patient_condition,
                    rejected_preference=gt_data[i-1]['preference']
                )

            while 1:
                # Obtain response from patient
                patient_kwargs.update(kwargs)
                patient_response = self.patient_agent(
                    self.dialog_history['scheduling'][-1]["content"],
                    using_multi_turn=True,
                    verbose=False,
                    **patient_kwargs,
                )
                self.dialog_history['scheduling'].append({"role": "Patient", "content": patient_response})
                role = f"{colorstr('green', 'Patient')} ({gt_patient_condition['preference']})"
                log(f"{role:<25}: {patient_response}")
                
                # Scheduling from staff
                staff_kwargs.update(kwargs)
                staff_known_condition.update({'patient_intention': patient_response})
                staff_response = self.scheduling(
                    client,
                    staff_known_condition,
                    doctor_information,
                    chat_history=self._to_lc_history('scheduling'),
                    **staff_kwargs
                )
                
                # Clarification message
                if staff_response['type'] == 'text':
                    response = staff_response['result']
                    self.dialog_history['scheduling'].append({"role": "Staff", "content": response})
                    role = f"{colorstr('blue', 'Staff')}"
                    log(f"{role:<25}: {response}")
                
                # Tool calling result
                elif staff_response['type'] == 'tool':
                    pred_schedule = staff_response['result']
                    response = self.admin_staff_agent.staff_suggestion.format(schedule=pred_schedule)
                    self.dialog_history['scheduling'].append({"role": "Staff", "content": response})
                    role = f"{colorstr('blue', 'Staff')}"
                    log(f"{role:<25}: {response}")
                    break
            
            # Sanity check
            if self.sanity_checker is None:
                status, status_code = True, STATUS_CODES['correct']
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
            if random.random() < preference_reject_prob and i != len(gt_data) - 1:
                preference_reject_prob *= self.preferene_rejection_prob_decay
            ## Non-rejection case
            else:
                self.dialog_history['scheduling'].append({"role": "Patient", "content": self.end_phrase})
                role = f"{colorstr('green', 'Patient')} ({gt_data[i]['preference']})"
                log(f"{role:<25}: {self.end_phrase}")
                break

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
                    'patient': staff_known_data[i]['patient'],
                    'attending_physician': pred_doctor_name,
                    'department': staff_known_data[i]['department'],
                    'date': schedule['date'],
                    'schedule': [schedule['start'], schedule['end']],
                    'patient_intention': staff_known_data[i]['patient_intention'],
                    'preference': gt_data[i].get('preference'),
                    'preferred_doctor': gt_data[i].get('preferred_doctor'),
                    'valid_from': gt_data[i].get('valid_from'),
                    'last_updated_time': self.environment.current_time
                }
                result_dict['pred'] = [prediction]
                result_dict['status'] = [True]
            except:
                log('No sanity checker is available; an error occurred while parsing the prediction. Returning a failure result.', level='warning')

        log("Simulation completed.", color=True)
        return doctor_information, result_dict

    
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
            gt_idx (Optional[int], optional): Ground-truth index of the appointment to be canceled. Defaults to None.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s).
            patient_schedules (Optional[list[dict]], optional): List of patient appointment schedules. Defaults to None.
            verbose (bool, optional): Whether to print conversation logs. Defaults to True.
            max_inferences (int, optional): Maximum number of dialogue turns.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff agent.
            **kwargs: Additional keyword arguments passed to the patient and staff agent.

        Returns:
            Tuple[dict, dict]: Updated doctor information and a result dictionary after cancellation.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, log(f"Doctor information must be provided if you don't use FHIR.", level="error")

        # Initialize agents and result dictionary
        result_dict = init_result_dict()
        self._init_agents(verbose=verbose)
        patient_schedules = self.environment.patient_schedules if patient_schedules is None else patient_schedules
        doctor_information = self.environment.get_general_doctor_info_from_fhir() if self.fhir_integration else doctor_information
        client = self.admin_staff_agent.build_agent(
            rule=self.rules, 
            doctor_info=doctor_information,
            patient_schedule_list=patient_schedules,
            gt_idx=gt_idx,
        )

        # Start conversation
        staff_greet = self.admin_staff_agent.general_staff_greet
        self.dialog_history['cancel'].append({"role": "Staff", "content": staff_greet})
        role = f"{colorstr('blue', 'Staff')}"
        log(f"{role:<25}: {staff_greet}")

        try:
            for _ in range(max_inferences):
                # Obtain response from patient
                patient_kwargs.update(kwargs)
                patient_response = self.patient_agent(
                    self.dialog_history['cancel'][-1]["content"],
                    using_multi_turn=True,
                    verbose=False,
                    **patient_kwargs
                )
                self.dialog_history['cancel'].append({"role": "Patient", "content": patient_response})
                role = f"{colorstr('green', 'Patient')} (cancel)"
                log(f"{role:<25}: {patient_response}")

                # Canceling from staff
                staff_response = self.canceling(
                    client=client,
                    patient_intention=patient_response,
                    chat_history=self._to_lc_history('cancel'),
                )
                # Clarification message instead of tool calling
                if staff_response['type'] == 'text':
                    response = staff_response['result']
                    self.dialog_history['cancel'].append({"role": "Staff", "content": response})
                    role = f"{colorstr('blue', 'Staff')}"
                    log(f"{role:<25}: {response}")
                    continue
                
                # Tool calling result
                elif staff_response['type'] == 'tool':
                    result = staff_response['result']
                    result_dict = result['result_dict']
                    
                    # Succesful case
                    if result_dict['status'][0] is not False:  # No GT and correct case
                        cancelled_schedule = {k: v for k, v in result['cancelled_schedule'].items() \
                                            if k in ['patient', 'attending_physician', 'department', 'date', 'schedule']}
                        
                        # Final response of staff
                        response = f"I've cancelled this schedule: {cancelled_schedule}"
                        self.dialog_history['cancel'].append({"role": "Staff", "content": response})
                        role = f"{colorstr('blue', 'Staff')}"
                        log(f"{role:<25}: {response}")

                        # Final response of patient
                        self.dialog_history['cancel'].append({"role": "Patient", "content": self.end_phrase})
                        role = f"{colorstr('green', 'Patient')} (cancel)"
                        log(f"{role:<25}: {self.end_phrase}")

                        result_dict['dialog'].append(preprocess_dialog(self.dialog_history['cancel']))
                        break
                    
                    # Fail to identify the schedule
                    else:
                        raise ScheduleNotFoundError(colorstr("red", "Error: Schedule not found error."))    
                
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
        except ScheduleNotFoundError:
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
            status_code = f"Unexpected error: {e}"
            log(status_code, level='warning')
            result_dict = {
                'gt': [{'cancel': gt_idx}],
                'pred': [None],
                'status': [False],
                'status_code': [status_code],
                'dialog': [preprocess_dialog(self.dialog_history['cancel'])]
            }

        log("Simulation completed.", color=True)
        return doctor_information, result_dict


    def rescheduling_simulate(self, 
                              gt_idx: Optional[int] = None,
                              doctor_information: Optional[dict] = None,
                              patient_schedules: Optional[list[dict]] = None,
                              verbose: bool = True,
                              max_inferences: int = 5,
                              patient_kwargs: dict = {},
                              staff_kwargs: dict = {},
                              **kwargs):
        """
        Simulate a multi-turn conversation for appointment rescheduling.

        Args:
            gt_idx (Optional[int], optional): Ground-truth index of the appointment to be canceled. Defaults to None.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s).
            patient_schedules (Optional[list[dict]], optional): List of patient appointment schedules. Defaults to None.
            verbose (bool, optional): Whether to print conversation logs. Defaults to True.
            max_inferences (int, optional): Maximum number of dialogue turns.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff agent.
            **kwargs: Additional keyword arguments passed to the patient and staff agents.

        Returns:
            int: Index of the rescheduled appointment, or -1 if rescheduling fails.

        Raises:
            TypeError: If the return type from the rescheduling method is unexpected.
        """
        # Initialize agents and result dictionary
        self._branch = False
        self.result_dict = init_result_dict()
        self._init_agents(verbose=verbose)
        patient_schedules = self.environment.patient_schedules if patient_schedules is None else patient_schedules
        doctor_information = self.environment.get_general_doctor_info_from_fhir() if self.fhir_integration else doctor_information
        client = self.admin_staff_agent.build_agent(
            rule=self.rules, 
            doctor_info=doctor_information,
            patient_schedule_list=patient_schedules,
            gt_idx=gt_idx,
        )

        # Start conversation
        staff_greet = self.admin_staff_agent.general_staff_greet
        self.dialog_history['reschedule'].append({"role": "Staff", "content": staff_greet})
        role = f"{colorstr('blue', 'Staff')}"
        log(f"{role:<25}: {staff_greet}")

        for _ in range(max_inferences):
            # Obtain response from patient
            patient_kwargs.update(kwargs)
            patient_response = self.patient_agent(
                self.dialog_history['reschedule'][-1]["content"],
                using_multi_turn=True,
                verbose=False,
                **patient_kwargs
            )
            self.dialog_history['reschedule'].append({"role": "Patient", "content": patient_response})
            role = f"{colorstr('green', 'Patient')} (move)"
            log(f"{role:<25}: {patient_response}")

            # Rescheduling from staff
            staff_kwargs.update(kwargs)
            staff_response = self.rescheduling(
                client=client,
                patient_intention=patient_response,
                chat_history=self._to_lc_history('reschedule'),
                doctor_information=doctor_information,
                **staff_kwargs
            )
            
            # Clarification message
            if staff_response['type'] == 'text':
                response = staff_response['result']
                self.dialog_history['reschedule'].append({"role": "Staff", "content": response})
                role = f"{colorstr('blue', 'Staff')}"
                log(f"{role:<25}: {response}")

            # Tool calling result
            elif staff_response['type'] == 'tool':
                result = staff_response['result']
                self._update_result_dict(result['result_dict'])
                if self.result_dict['status'][0] is not False:  # No GT and correct case
                    original_schedule = result['original_schedule']
                    new_schedule = result['new_schedule']
                    
                    # Save point
                    self.result_dict['dialog'] = [preprocess_dialog(self.dialog_history['reschedule'])]
                    yield {'type': 'simulation', 'original': original_schedule, 'prediction': new_schedule}

                    if self._branch:
                        pred_idx = self.result_dict['pred'][0]['reschedule']
                        pred_doctor_name = list(new_schedule['schedule'].keys())[0]
                        old_iso_time = get_iso_time(original_schedule['schedule'][0], original_schedule['date'])
                        new_iso_time = get_iso_time(new_schedule['schedule'][pred_doctor_name]['start'], new_schedule['schedule'][pred_doctor_name]['date'])
                        
                        if compare_iso_time(old_iso_time, new_iso_time):
                            self._updated_doctor_information = self.rules.cancel_schedule(pred_idx, doctor_information, original_schedule)
                            prediction = {
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
                                'last_updated_time': self.environment.current_time
                            }

                            tmp_original_schedule = {k: v for k, v in original_schedule.items() \
                                                     if k in ['patient', 'attending_physician', 'department', 'date', 'schedule']}
                            tmp_prediction_schedule = {k: v for k, v in prediction.items() \
                                                       if k in ['patient', 'attending_physician', 'department', 'date', 'schedule']}
                            
                            response = f"I've moved your original schedule: {tmp_original_schedule} to the new one: {tmp_prediction_schedule}"
                            
                        else:
                            tmp_original_schedule = {k: v for k, v in original_schedule.items() \
                                                     if k in ['patient', 'attending_physician', 'department', 'date', 'schedule']}
                            response = f"There are no available times. I've added this schedule to the waiting list: {tmp_original_schedule}"
                            self.environment.add_waiting_list(pred_idx, verbose)
                            self._branch = False

                        #  Final response of staff
                        self.dialog_history['reschedule'].append({"role": "Staff", "content": response})
                        role = f"{colorstr('blue', 'Staff')}"
                        log(f"{role:<25}: {response}")

                        # Final response of patient
                        self.dialog_history['reschedule'].append({"role": "Patient", "content": self.end_phrase})
                        role = f"{colorstr('green', 'Patient')} (move)"
                        log(f"{role:<25}: {self.end_phrase}")
                    
                    break
                
                else:
                    self.result_dict['dialog'].append(preprocess_dialog(self.dialog_history['reschedule']))
                    raise ScheduleNotFoundError(colorstr("red", "Error: Schedule not found error."))
            
            else:
                self.result_dict['dialog'].append(preprocess_dialog(self.dialog_history['reschedule']))
                raise TypeError(colorstr("red", "Error: Unexpected return type from rescheduling method."))
            
        # End of conversation
        self.result_dict['dialog'] = ['\n'.join([f"{turn['role']}: {' '.join(turn['content'].split())}" for turn in self.dialog_history['reschedule']])]
        log("Simulation completed.", color=True)

        if self._branch:
            yield {'type': 'update', 'original': original_schedule, 'prediction': prediction}
    