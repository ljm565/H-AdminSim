import re
import os
import json
import random
from copy import deepcopy
from importlib import resources
from patientsim import PatientAgent
from decimal import Decimal, getcontext
from typing import Tuple, Union, Optional
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

from h_adminsim import SchedulingAdminStaffAgent
from h_adminsim.registry.errors import ToolCallingError, DataNotFoundError, SchedulingError
from h_adminsim.registry import OPFU_PREFERENCE_PHRASE_PATIENT, OPFU_PREFERENCE_PHRASE_STAFF, STATUS_CODES
from h_adminsim.environment.hospital import HospitalEnvironment
from h_adminsim.utils import log, colorstr
from h_adminsim.tools.callback import TokenUsageCallback
from h_adminsim.tools.sanity_checker import SanityChecker
from h_adminsim.tools import SchedulingRule, scheduling_tool_calling
from h_adminsim.utils.common_utils import *



class OPFUSchedulingSimulation:
    def __init__(self,
                 patient_agent: PatientAgent,
                 admin_staff_agent: SchedulingAdminStaffAgent,
                 metadata: dict,
                 department_data: dict,
                 environment: HospitalEnvironment,
                 scheduling_strategy: str = 'tool_calling',
                 preference_rejection_prob: float = 0.3,
                 preference_rejection_prob_decay: float = 0.5,
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
        self.scheduling_strategy = scheduling_strategy
        self.preference_rejection_prob = preference_rejection_prob
        self.preference_rejection_prob_decay = preference_rejection_prob_decay
        self.fhir_integration = fhir_integration
        self._init_prompt(schedule_rejection_prompt_path)
        self.sanity_checker = sanity_checker
        self.rules = SchedulingRule(metadata, department_data, self.environment, self.fhir_integration)
        
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
        self._init_history()

    
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
            prompt_file_name = "opfu_schedule_patient_rejected_system.txt"
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            self.rejection_system_prompt_template = file_path.read_text()
        
        # User can specify a custom system prompt
        else:
            if not os.path.exists(schedule_rejection_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {schedule_rejection_prompt_path}"))
            with open(schedule_rejection_prompt_path, 'r') as f:
                self.rejection_system_prompt_template = f.read()


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
            'test_scheduling': [],
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
                        text_dict = json.loads(match.group(1))
                    else:
                        try:
                            text_dict = json.loads(data)
                        except:
                            return data
                else:
                    text_dict = data

                assert {'test_schedule', 'all_results_ready_at'} <= set(text_dict.keys())
                assert isinstance(text_dict['test_schedule'], list)
                for entry in text_dict['test_schedule']:
                    assert isinstance(entry, dict) and len(entry) == 1
                    dev = next(iter(entry))
                    entry[dev]['start'] = float(entry[dev]['start'])
                    entry[dev]['end']   = float(entry[dev]['end'])
                    entry[dev]['date']  = str(entry[dev]['date'])
                text_dict['all_results_ready_at'] = str(text_dict['all_results_ready_at'])

                # Optional follow-up consultation slot
                fu = text_dict.get('fu_schedule')
                if isinstance(fu, str) and fu.lower() == 'none':
                    fu = None
                if isinstance(fu, dict) and fu:
                    doctor = next(iter(fu))
                    assert isinstance(fu[doctor], dict)
                    fu[doctor]['start'] = float(fu[doctor]['start'])
                    fu[doctor]['end']   = float(fu[doctor]['end'])
                    fu[doctor]['date']  = str(fu[doctor]['date'])
                    text_dict['fu_schedule'] = fu
                else:
                    text_dict['fu_schedule'] = None
                return text_dict

            except:
                return str(data)
        
        elif strategy == 'tool_calling':
            schedule = {'test_schedule': [], 'fu_schedule': None, 'all_results_ready_at': data['all_results_ready_at']}
            
            # Follow-up visit schedule post-processing
            fu_schedule = data['fu_schedule']
            if fu_schedule['doctor'] and fu_schedule['schedule']:
                doctor = fu_schedule['doctor'][0]
                duration = filtered_doctor_information['doctor'][doctor]['outpatient_duration']
                date, st_hour = iso_to_date(fu_schedule['schedule'][0]), iso_to_hour(fu_schedule['schedule'][0])
                tr_hour = float(Decimal(str(duration)) + Decimal(str(st_hour)))
                schedule['fu_schedule'] = {doctor: {'date': date, 'start': st_hour, 'end': tr_hour}}
            
            # Test post-processing
            required_tests = data['tests']
            for _, values in required_tests.items():
                device_code, date = values['device'], values['date']
                st_hour, tr_hour = iso_to_hour(values['start']), iso_to_hour(values['end'])
                schedule['test_schedule'].append(
                    {device_code: {'date': date, 'start': st_hour, 'end': tr_hour}}
                )
            return schedule


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
            # Build new system prompts for rejection scenario.
            # OPFU preferences are 'asap' and 'batch' only (no doctor / date).
            preference = patient_condition.get('preference')
            preference_desc = OPFU_PREFERENCE_PHRASE_PATIENT[preference]
            rejected_preference_desc = OPFU_PREFERENCE_PHRASE_STAFF[rejected_preference]
            system_prompt = self.rejection_system_prompt_template.format(
                preference=preference,
                preference_desc=preference_desc,
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


    def test_scheduling(self,
                        client: AgentExecutor,
                        known_condition: dict,
                        doctor_information: Optional[dict] = None, 
                        test_device_information: Optional[dict] = None, 
                        reschedule_flag: bool = False,
                        chat_history: list = [],
                        reasoning_max_tries: int = 0,
                        **kwargs) -> dict:
        """
        Make a test appointment between the doctor and the patient.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            known_condition (dict): Patient conditions known to the staff.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details. Defaults to None.
            test_device_information (Optional[dict], optional): A dictionary containing information about the test devices available. Defaults to None.
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
            assert test_device_information is not None, colorstr("red", f"Test device information must be provided if you don't use FHIR.")

        # Initialization based on the known condition from the staff
        result_dict = init_result_dict()
        callback = kwargs.pop('callback', None)
        department = known_condition['department']
        filtered_doctor_information = self.environment.get_doctor_schedule(
            doctor_information=doctor_information,
            department=department,
            fhir_integration=self.fhir_integration and doctor_information is None,
        )
        
        # First, try to use the tool calling
        try:
            assert self.scheduling_strategy == 'tool_calling', log('Scheduling strategy is set to `reasoning`, directly use the reasoning method.', level='warning')
            
            # Invoke
            prediction = scheduling_tool_calling(
                client=client, 
                user_prompt=known_condition['patient_intention'],
                history=chat_history,
                callback=callback,
            )

            # Scheduling result
            if prediction['type'] == 'tool':
                res = prediction['result']
                if res['action'] == 'retrieval':
                    st = res['status']
                    idx = res['index']

                    # Patient information not found case: -> text
                    if st is None and idx['pred'] == -1:
                        prediction['type'] = 'text'
                        prediction['result'] = "Sorry, we couldn't find a matching information. Could you please check your details again?"
                        return prediction

                    if st is None:  # No GT, retrieved
                        result_dict['gt'].append({'test_retrieve': None})
                        result_dict['pred'].append({'test_retrieve': idx['pred']})
                        result_dict['status'].append(None)
                        result_dict['status_code'].append(None)
                    elif st is False:  # GT exists, identification failed
                        result_dict['gt'].append({'test_retrieve': idx['gt']})
                        result_dict['pred'].append({'test_retrieve': idx['pred']})
                        result_dict['status'].append(False)
                        result_dict['status_code'].append(STATUS_CODES['test_retrieve']['identify'])
                        prediction['tmp_flag'] = 'retrieve'
                    else:  # True
                        result_dict['gt'].append({'test_retrieve': idx['gt']})
                        result_dict['pred'].append({'test_retrieve': idx['pred']})
                        result_dict['status'].append(True)
                        result_dict['status_code'].append(STATUS_CODES['correct'])

                    prediction['result_dict'] = result_dict
                    return prediction
                
                elif res['action'] == 'scheduling':
                    test_schedules = OPFUSchedulingSimulation.postprocessing(
                        strategy='tool_calling',
                        data=prediction['result'],
                        filtered_doctor_information=filtered_doctor_information,
                    )
                    prediction['result'] = test_schedules

                else:
                    raise ToolCallingError(colorstr('red', 'Failed to choose an appropriate scheduling tool.'))
            
            ## Dialogue
            elif prediction['type'] == 'text':
                if 'no tool' in prediction['result'].lower():
                    raise ToolCallingError(colorstr('red', 'Failed to choose an appropriate scheduling tool.'))
            
            ## Error
            else:
                raise TypeError(colorstr("red", "Error: Unexpected return type from scheduling method."))

        # If tool calling fails, fallback to LLM-based scheduling
        except ToolCallingError:
            # Fallback only applies after retrieve_patient_tests has populated required_tests.
            if not known_condition.get('required_tests'):
                raise

            if self.scheduling_strategy == 'tool_calling':
                log('Failed to select an appropriate tool. Falling back to reasoning-based scheduling.', level='warning')

            required_test_codes = [t['test_code'] for t in known_condition['required_tests']]
            filtered_test_devices = self.environment.get_test_device_schedule(
                test_device_information=test_device_information if not self.fhir_integration else None,
                test_code=required_test_codes,
                fhir_integration=self.fhir_integration,
                express_detail=True,
            )
            user_prompt = self.admin_staff_agent.scheduling_user_prompt_template.format(
                START_HOUR=self._START_HOUR,
                END_HOUR=self._END_HOUR,
                TIME_UNIT=self._TIME_UNIT,
                CURRENT_TIME=self.environment.current_time,
                UTC_OFFSET=self.environment._utc_offset,
                DEPARTMENT=department,
                ATTENDING_PHYSICIAN=known_condition.get('attending_physician', ''),
                PREFERENCE=known_condition['patient_intention'],
                DAY=self._DAY,
                TESTS=json.dumps(known_condition['required_tests'], indent=2),
                TEST_DEVICES=json.dumps(filtered_test_devices, indent=2),
                DOCTOR_SCHEDULES=json.dumps(filtered_doctor_information, indent=2),
            )

            tries, schedule = 0, None
            while True:
                schedule = self.admin_staff_agent(
                    user_prompt,
                    using_multi_turn=False,
                    verbose=False,
                    **kwargs,
                )
                schedule = OPFUSchedulingSimulation.postprocessing(
                    strategy='reasoning',
                    data=schedule,
                )
                if isinstance(schedule, dict) or tries >= reasoning_max_tries:
                    break
                tries += 1

            prediction = {
                'type': 'tool',
                'result': schedule if isinstance(schedule, dict) else {'test_schedule': [], 'fu_schedule': None, 'all_results_ready_at': ''},
                'raw': None,
                'token': deepcopy(self.admin_staff_agent.client.token_usages),
            }
            self.admin_staff_agent.reset_history(verbose=False)

        return prediction
    

    def test_scheduling_simulate(self,
                                 gt_data: dict,
                                 staff_known_data: dict,
                                 doctor_information: Optional[dict] = None,
                                 test_device_information: Optional[dict] = None,
                                 verbose: bool = False,
                                 max_inferences: int = 8,
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
            test_device_information (Optional[dict], optional): A dictionary containing information about the test devices available. Defaults to None.
            verbose (bool, optional): Whether to log detailed simulation outputs. Defaults to False.
            max_inferences (int, optional): Maximum number of dialogue turns.
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
            assert test_device_information is not None, colorstr("red", f"Test device information must be provided if you don't use FHIR.")

        # Initialize agents and result dictionary
        staff_token_callback = TokenUsageCallback()
        self._init_agents(verbose=verbose)
        staff_token_stats = {}
        patient_info = self.environment.patient_schedules
        client = self.admin_staff_agent.build_agent(
            rule=self.rules, 
            doctor_info=None,
            patient_schedule_list=patient_info,
            gt_idx=gt_data[0]['index'],
        )
        merged_patient_kwargs = {**patient_kwargs, **kwargs}
        merged_staff_kwargs = {**staff_kwargs, **kwargs}
        
        # Start conversation
        staff_greet = self.admin_staff_agent.general_greet
        self.dialog_history['test_scheduling'].append({"role": "Staff", "content": staff_greet})
        role = f"{colorstr('blue', 'Staff')}"
        log(f"{role:<25}: {staff_greet}")

        # Iterate over multiple preferences if exists
        tries = 0
        preference_reject_prob = 0.0 if len(gt_data) <= 1 else self.preference_rejection_prob
        try:
            # Preference iteration
            for i, gt_patient_condition in enumerate(gt_data):
                # For the rejection scenario
                if i != 0:
                    self.update_patient_system_prompt(
                        patient_condition=gt_patient_condition,
                        rejected_preference=gt_data[i-1]['preference']
                    )

                while 1:
                    # Obtain response from patient
                    patient_response = self.patient_agent(
                        self.dialog_history['test_scheduling'][-1]["content"],
                        using_multi_turn=True,
                        verbose=False,
                        **merged_patient_kwargs,
                    )
                    patient_token_stats = self.patient_agent.client.token_usages
                    self.dialog_history['test_scheduling'].append({"role": "Patient", "content": patient_response})
                    role = f"{colorstr('green', 'Patient')} ({gt_patient_condition['preference']})"
                    log(f"{role:<25}: {patient_response}")
                    
                    # Scheduling from staff
                    staff_known_data.update({'patient_intention': patient_response})
                    staff_response = self.test_scheduling(
                        client,
                        staff_known_data,
                        doctor_information,
                        test_device_information,
                        chat_history=self._to_lc_history('test_scheduling'),
                        reasoning_max_tries=reasoning_max_tries,
                        callback=staff_token_callback,
                        **merged_staff_kwargs
                    )

                    # Update token stats
                    if self.scheduling_strategy == 'tool_calling':
                        staff_token_stats = staff_token_callback.token_usage
                    else:
                        for k, v in staff_response['token'].items():
                            if k not in staff_token_stats:
                                staff_token_stats[k] = deepcopy(v)
                            else:
                                staff_token_stats[k].extend(v)
                    
                    # Clarification message
                    if staff_response['type'] == 'text':
                        response = staff_response['result']
                        self.dialog_history['test_scheduling'].append({"role": "Staff", "content": response})
                        role = f"{colorstr('blue', 'Staff')}"
                        log(f"{role:<25}: {response}")
                    
                    # Tool calling result
                    elif staff_response['type'] == 'tool':
                        # Fail to identify the schedule
                        if staff_response.get('tmp_flag') == 'retrieve':
                            result_dict = staff_response['result_dict']
                            raise DataNotFoundError(colorstr("red", "Error: Patient information not found error."))
                        
                        # Successful case
                        else:
                            # Test retrieval case
                            if 'test_list' in staff_response['result']:
                                result = staff_response['result']
                                _patient_info = result['patient_fv']
                                test_list = result['test_list']
                                staff_known_data.update({'patient_fv': _patient_info})
                                staff_known_data.update({'department': _patient_info['department']})
                                staff_known_data.update({'attending_physician': _patient_info['attending_physician']})
                                staff_known_data.update({'required_tests': test_list})
                                test_list_desc = ', '.join([t['name'] for t in test_list])
                                required_test_codes = [_test['test_code'] for _test in test_list]
                                filtered_doctor_information = self.environment.get_doctor_schedule(
                                    doctor_information=doctor_information,
                                    department=staff_known_data['department'],
                                    fhir_integration=self.fhir_integration and doctor_information is None,
                                )
                                filtered_test_device_information = self.environment.get_test_device_schedule(
                                    test_device_information=test_device_information,
                                    test_code=required_test_codes,
                                    fhir_integration=self.fhir_integration and test_device_information is None,
                                )

                                # Rebuild the staff agent so subsequent turns have the test-scheduling tools.
                                client = self.admin_staff_agent.build_agent(
                                    rule=self.rules,
                                    doctor_info=filtered_doctor_information,
                                    patient_schedule_list=patient_info,
                                    gt_idx=gt_data[0]['index'],
                                    filtered_test_device_information=filtered_test_device_information,
                                    required_test_codes=required_test_codes,
                                )

                                # Response formatting for test guidance
                                try:
                                    if natural_express:
                                        _format = random.choice(self.admin_staff_agent.natural_test_explanation) \
                                            if isinstance(self.admin_staff_agent.natural_test_explanation, list) \
                                                else self.admin_staff_agent.natural_test_explanation
                                        response = _format.format(
                                            test_len=len(test_list),
                                            test_list=test_list_desc
                                        ) + ' ' + self.admin_staff_agent.test_greet
                                    else:
                                        response = self.admin_staff_agent.test_explanation.format(
                                            test_len=len(test_list),
                                            test_list=test_list_desc
                                        ) + ' ' + self.admin_staff_agent.test_greet
                                except:
                                    try:
                                        response = self.admin_staff_agent.test_explanation.format(
                                            test_len=len(test_list),
                                            test_list=test_list_desc
                                        ) + ' ' + self.admin_staff_agent.test_greet
                                    except:
                                        response = 'Your tests: ' + str(test_list_desc) + ' ' + self.admin_staff_agent.test_greet
                            
                            elif 'test_schedule' in staff_response['result']:
                                pred_schedule  = staff_response['result']
                                pred_schedules = pred_schedule['test_schedule']

                                # Build a humanized summary of every test slot
                                parts = []
                                for entry in pred_schedules:
                                    for dev_code, slot in entry.items():
                                        parts.append(
                                            f"{dev_code} on {slot['date']} from {slot['start']} to {slot['end']}"
                                        )
                                fu_slot = pred_schedule.get('fu_schedule')
                                if isinstance(fu_slot, dict) and fu_slot:
                                    fu_doctor = next(iter(fu_slot))
                                    fu_info = fu_slot[fu_doctor]
                                    parts.append(
                                        f"follow-up with {fu_doctor} on {fu_info['date']} from {fu_info['start']} to {fu_info['end']}"
                                    )
                                schedule_summary = "; ".join(parts)

                                # Response formatting for test guidance
                                try:
                                    if natural_express:
                                        _format = random.choice(self.admin_staff_agent.natural_fu_schedule_suggestion) \
                                            if isinstance(self.admin_staff_agent.natural_fu_schedule_suggestion, list) \
                                                else self.admin_staff_agent.natural_fu_schedule_suggestion
                                        response = _format.format(schedule_summary=schedule_summary)
                                    else:
                                        response = self.admin_staff_agent.fu_schedule_suggestion.format(
                                            schedule_summary=schedule_summary
                                        )
                                except:
                                    response = 'Your test schedules: ' + schedule_summary

                            self.dialog_history['test_scheduling'].append({"role": "Staff", "content": response})
                            role = f"{colorstr('blue', 'Staff')}"
                            log(f"{role:<25}: {response}")

                            # A successful test schedule ends the inner dialog loop.
                            if 'test_schedule' in staff_response['result']:
                                break

                    tries += 1
                    if tries > max_inferences:
                        result_dict = {
                            'gt': [gt_patient_condition],
                            'pred': [None],
                            'status': [False],
                            'status_code': [STATUS_CODES['simulation']],
                            'dialog': [preprocess_dialog(self.dialog_history['test_scheduling'])]
                        }
                        token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
                        return doctor_information, result_dict, token_usage

                # Sanity check
                ## No GT case
                if self.sanity_checker is None:
                    status, status_code = True, STATUS_CODES['correct']
                ## GT existing case
                else:
                    status, status_code = self.sanity_checker.test_schedule_check(
                        prediction=pred_schedule,
                        gt_patient_condition=gt_patient_condition,
                        test_device_information=filtered_test_device_information,
                        doctor_information=filtered_doctor_information,
                        environment=self.environment,
                        rule=self.rules,
                    )

                if not status:
                    break

                # Preference rejection logic (OPFU: asap vs batch)
                next_pref_differs = (i != len(gt_data) - 1) and \
                    (gt_data[i + 1]['preference'] != gt_data[i]['preference'])
                if random.random() < preference_reject_prob and next_pref_differs:
                    preference_reject_prob *= self.preference_rejection_prob_decay
                ## Non-rejection case
                else:
                    if natural_express:
                        self.update_patient_system_prompt(
                            new_system_prompt=self.patient_satisfaction_system_prompt
                        )
                        patient_response = self.patient_agent(
                            self.natural_end_phrase.format(schedule=self.dialog_history['test_scheduling'][-1]['content']),
                            using_multi_turn=True,
                            verbose=False,
                            **merged_patient_kwargs,
                        )
                        patient_token_stats = self.patient_agent.client.token_usages

                    else:
                        patient_response = self.end_phrase

                    self.dialog_history['test_scheduling'].append({"role": "Patient", "content": patient_response})
                    role = f"{colorstr('green', 'Patient')} ({gt_data[i]['preference']})"
                    log(f"{role:<25}: {patient_response}")

                    break

        except DataNotFoundError:
            result_dict['dialog'].append(preprocess_dialog(self.dialog_history['test_scheduling']))
            log("Simulation completed.", color=True)
            token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
            return doctor_information, result_dict, token_usage

        # Organize the result for the regular success / failure path
        result_dict = {
            'gt': [gt_patient_condition],
            'pred': [pred_schedule],
            'status': [False],
            'status_code': [status_code],
            'dialog': [preprocess_dialog(self.dialog_history['test_scheduling'])]
        }
        if status:
            try:
                prediction = {
                    'patient_fv': staff_known_data.get('patient_fv'),
                    'department': staff_known_data.get('department'),
                    'attending_physician': staff_known_data.get('attending_physician'),
                    'required_tests': staff_known_data.get('required_tests'),
                    'test_schedule': pred_schedule['test_schedule'],
                    'fu_schedule': pred_schedule.get('fu_schedule'),
                    'all_results_ready_at': pred_schedule['all_results_ready_at'],
                    'patient_intention': staff_known_data.get('patient_intention'),
                    'preference': gt_data[i].get('preference'),
                    'last_updated_time': self.environment.current_time,
                }
                result_dict['pred'] = [prediction]
                result_dict['status'] = [True]
            except Exception:
                result_dict['status_code'] = [STATUS_CODES['format']]
                log('Error while organizing the prediction. Returning a failure result.', level='warning')

        log("Simulation completed.", color=True)
        token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}

        return doctor_information, test_device_information, result_dict, token_usage


    # def scheduling_simulate_stream(self,
    #                                gt_data: dict,
    #                                staff_known_data: dict,
    #                                doctor_information: Optional[dict] = None,
    #                                verbose: bool = False,
    #                                max_inferences: int = 5,
    #                                natural_express: bool = True,
    #                                reasoning_max_tries: int = 3,
    #                                patient_kwargs: dict = {},
    #                                staff_kwargs: dict = {},
    #                                **kwargs):
    #     """
    #     Simulate a multi-turn outpatient scheduling dialogue between a patient agent and an administrative staff agent.

    #     Args:
    #         gt_data (dict): Ground-truth patient condition(s) for each dialogue turn.
    #         staff_known_data (dict): Patient information known to the staff agent.
    #         doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
    #                                                        including availability and other relevant details. Defaults to None.
    #         verbose (bool, optional): Whether to log detailed simulation outputs. Defaults to False.
    #         max_inferences (int, optional): Maximum number of dialogue turns.
    #         natural_express (bool, optional): Whether express new schedule as natural or not. Defaults to True.
    #         reasoning_max_tries (int, optional): Reasoning fallback maximum number of retries. Defaults to 3.
    #         patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
    #         staff_kwargs (dict, optional): Additional keyword arguments passed to the staff scheduling function.
    #         **kwargs: Shared keyword arguments passed to both agents.
    #     """
    #     # Sanity Check
    #     if not self.fhir_integration:
    #         assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

    #     # Initialize agents and result dictionary
    #     staff_token_callback = TokenUsageCallback()
    #     self._init_agents(verbose=verbose)
    #     staff_token_stats = {}
    #     filtered_doctor_information = self.environment.get_doctor_schedule(
    #         doctor_information=doctor_information if not self.fhir_integration else None,
    #         department=staff_known_data['department'],
    #         fhir_integration=self.fhir_integration,
    #     )
    #     client = self.admin_staff_agent.build_agent(
    #         rule=self.rules, 
    #         doctor_info=filtered_doctor_information
    #     )
    #     merged_patient_kwargs = {**patient_kwargs, **kwargs}
    #     merged_staff_kwargs = {**staff_kwargs, **kwargs}
        
    #     # Start conversation
    #     staff_greet = self.admin_staff_agent.appn_greet
    #     self.dialog_history['test_scheduling'].append({"role": "Staff", "content": staff_greet})
    #     role = f"{colorstr('blue', 'Staff')}"
    #     log(f"{role:<25}: {staff_greet}")

    #     # Iterate over multiple preferences if exists
    #     preference_reject_prob = 0.0 if len(gt_data) <= 1 else self.preference_rejection_prob
    #     for i, gt_patient_condition in enumerate(gt_data):
    #         # For the rejection scenario
    #         if i != 0:
    #             self.update_patient_system_prompt(
    #                 patient_condition=gt_patient_condition,
    #                 rejected_preference=gt_data[i-1]['preference']
    #             )

    #         tries = 0
    #         while 1:
    #             # Obtain response from patient
    #             patient_response = run_with_retry(
    #                 self.patient_agent,
    #                 self.dialog_history['test_scheduling'][-1]["content"],
    #                 using_multi_turn=True,
    #                 verbose=False,
    #                 max_retries=5,
    #                 **merged_patient_kwargs,
    #             )
    #             self.dialog_history['test_scheduling'].append({"role": "Patient", "content": patient_response})
    #             role = f"{colorstr('green', 'Patient')} ({gt_patient_condition['preference']})"
    #             log(f"{role:<25}: {patient_response}")
    #             yield 'Patient', preprocess_utterance(patient_response), None
                
    #             # Scheduling from staff
    #             staff_known_data.update({'patient_intention': patient_response})
    #             staff_response = run_with_retry(
    #                 self.scheduling,
    #                 client,
    #                 staff_known_data,
    #                 doctor_information,
    #                 chat_history=self._to_lc_history('test_scheduling'),
    #                 reasoning_max_tries=reasoning_max_tries,
    #                 callback=staff_token_callback,
    #                 **merged_staff_kwargs
    #             )
    #             if self.scheduling_strategy == 'tool_calling':
    #                 staff_token_stats = staff_token_callback.token_usage
    #             else:
    #                 for k, v in staff_response['token'].items():
    #                     if k not in staff_token_stats:
    #                         staff_token_stats[k] = deepcopy(v)
    #                     else:
    #                         staff_token_stats[k].extend(v)  # 두 번째~: extend
                
    #             # Clarification message
    #             if staff_response['type'] == 'text':
    #                 response = staff_response['result']
    #                 self.dialog_history['test_scheduling'].append({"role": "Staff", "content": response})
    #                 role = f"{colorstr('blue', 'Staff')}"
    #                 log(f"{role:<25}: {response}")
    #                 yield 'Staff', preprocess_utterance(response), None
                
    #             # Tool calling result
    #             elif staff_response['type'] == 'tool':
    #                 pred_schedule = staff_response['result']
                    
    #                 # Response formatting
    #                 try:
    #                     if natural_express:
    #                         _schedule = pred_schedule['schedule']
    #                         _doctor = list(_schedule.keys())[0]
    #                         date, st, tr = _schedule[_doctor]['date'], _schedule[_doctor]['start'], _schedule[_doctor]['end']
    #                         _format = random.choice(self.admin_staff_agent.natural_schedule_suggestion) \
    #                             if isinstance(self.admin_staff_agent.natural_schedule_suggestion, list) \
    #                                 else self.admin_staff_agent.natural_schedule_suggestion
    #                         response = _format.format(
    #                             doctor=_doctor, date=date, start=st, end=tr
    #                         )
    #                     else:
    #                         response = self.admin_staff_agent.schedule_suggestion.format(schedule=pred_schedule)
    #                 except:
    #                     try:
    #                         response = self.admin_staff_agent.schedule_suggestion.format(schedule=pred_schedule)
    #                     except:
    #                         response = str(pred_schedule)
                    
    #                 self.dialog_history['test_scheduling'].append({"role": "Staff", "content": response})
    #                 role = f"{colorstr('blue', 'Staff')}"
    #                 log(f"{role:<25}: {response}")
    #                 yield 'Staff', preprocess_utterance(response), pred_schedule
    #                 break

    #             tries += 1
    #             if tries > max_inferences:
    #                 return
            
    #         # Preference rejection logic
    #         ## Rejection case
    #         _schedule = pred_schedule['schedule']
    #         _doctor = list(_schedule.keys())[0]
    #         _date = _schedule[_doctor]['date']
    #         if random.random() < preference_reject_prob and i != len(gt_data) - 1 and \
    #             gt_data[i+1]['preferred_doctor'] != _doctor and gt_data[i+1]['valid_from'] != _date:  # Avoid overlap with next preferred doctor or next preferred date
    #             preference_reject_prob *= self.preference_rejection_prob_decay
            
    #         ## Non-rejection case
    #         else:
    #             if natural_express:
    #                 final_preference = gt_patient_condition.get('preference')
    #                 final_preferred_condition = gt_patient_condition.get('valid_from') if final_preference == 'date' \
    #                     else gt_patient_condition.get('preferred_doctor')

    #                 if final_preference.lower() == 'asap':
    #                     self.update_patient_system_prompt(
    #                         new_system_prompt=self.patient_satisfaction_system_prompt
    #                     )
    #                     patient_response = run_with_retry(
    #                         self.patient_agent,
    #                         self.natural_end_phrase.format(schedule=self.dialog_history['test_scheduling'][-1]['content']),
    #                         using_multi_turn=True,
    #                         verbose=False,
    #                         max_retries=5,
    #                         **merged_patient_kwargs,
    #                     )
    #                     self.dialog_history['test_scheduling'].append({"role": "Patient", "content": patient_response})
    #                     role = f"{colorstr('green', 'Patient')} ({gt_data[i]['preference']})"
    #                     log(f"{role:<25}: {patient_response}")
    #                     yield 'Patient', preprocess_utterance(patient_response), None

    #                 else:
    #                     # doctor or date preference - evaluate with retry
    #                     accept_tries = 0
    #                     while accept_tries <= max_inferences:
    #                         self.update_patient_system_prompt(
    #                             new_system_prompt=self.patient_evaluation_system_prompt.format(
    #                                 preference=final_preference,
    #                                 preferred_condition=final_preferred_condition
    #                             )
    #                         )
    #                         eval_phrase = self.patient_schedule_evaluation_phrase.format(
    #                             schedule=self.dialog_history['test_scheduling'][-1]['content'],
    #                             preference=final_preference,
    #                             preferred_condition=final_preferred_condition
    #                         )

    #                         patient_response = run_with_retry(
    #                             self.patient_agent,
    #                             eval_phrase,
    #                             using_multi_turn=True,
    #                             verbose=False,
    #                             max_retries=5,
    #                             **merged_patient_kwargs,
    #                         )

    #                         self.dialog_history['test_scheduling'].append({"role": "Patient", "content": patient_response})
    #                         role = f"{colorstr('green', 'Patient')} ({gt_data[i]['preference']})"
    #                         log(f"{role:<25}: {patient_response}")
    #                         yield 'Patient', preprocess_utterance(patient_response), None

    #                         if '#ACCEPT' in patient_response:
    #                             break

    #                         # Not accepted - retry tool calling
    #                         staff_known_data.update({'patient_intention': patient_response})
    #                         staff_response = run_with_retry(
    #                             self.scheduling,
    #                             client,
    #                             staff_known_data,
    #                             doctor_information,
    #                             chat_history=self._to_lc_history('test_scheduling'),
    #                             reasoning_max_tries=reasoning_max_tries,
    #                             callback=staff_token_callback,
    #                             **merged_staff_kwargs
    #                         )
    #                         if staff_response['type'] == 'tool':
    #                             pred_schedule = staff_response['result']
    #                             _schedule = pred_schedule['schedule']
    #                             _doctor = list(_schedule.keys())[0]
    #                             _date = _schedule[_doctor]['date']
    #                             try:
    #                                 date, st, tr = _schedule[_doctor]['date'], _schedule[_doctor]['start'], _schedule[_doctor]['end']
    #                                 _format = random.choice(self.admin_staff_agent.natural_schedule_suggestion) \
    #                                     if isinstance(self.admin_staff_agent.natural_schedule_suggestion, list) \
    #                                         else self.admin_staff_agent.natural_schedule_suggestion
    #                                 response = _format.format(doctor=_doctor, date=date, start=st, end=tr)
    #                             except:
    #                                 try:
    #                                     response = self.admin_staff_agent.schedule_suggestion.format(schedule=pred_schedule)
    #                                 except:
    #                                     response = str(pred_schedule)
    #                             self.dialog_history['test_scheduling'].append({"role": "Staff", "content": response})
    #                             role = f"{colorstr('blue', 'Staff')}"
    #                             log(f"{role:<25}: {response}")
    #                             yield 'Staff', preprocess_utterance(response), pred_schedule
                            
    #                         elif staff_response['type'] == 'text':
    #                             response = staff_response['result']
    #                             self.dialog_history['test_scheduling'].append({"role": "Staff", "content": response})
    #                             role = f"{colorstr('blue', 'Staff')}"
    #                             log(f"{role:<25}: {response}")
    #                             yield 'Staff', preprocess_utterance(response), None

    #                         accept_tries += 1

    #             else:
    #                 patient_response = self.end_phrase
    #                 self.dialog_history['test_scheduling'].append({"role": "Patient", "content": patient_response})
    #                 role = f"{colorstr('green', 'Patient')} ({gt_data[i]['preference']})"
    #                 log(f"{role:<25}: {patient_response}")
    #                 yield 'Patient', preprocess_utterance(patient_response), None

    #             break
