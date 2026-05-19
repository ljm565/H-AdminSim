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
                       filtered_doctor_information: Optional[dict] = None,
                       required_tests: Optional[list] = None,
                       filtered_test_device_information: Optional[dict] = None,
                       utc_offset: Optional[str] = None,
                       rule: Optional[SchedulingRule] = None,
                       attending_physician: Optional[str] = None) -> Union[str, dict]:
        """
        Attempts to parse the given text as JSON. If parsing succeeds, returns a dictionary;
        otherwise, returns the original string.

        Args:
            strategy (str): Scheduling strategy. It must be either `reasoning` or `tool_calling`.
            data (Union[str, dict]): The text output to post-process, potentially a JSON-formatted string.
            filtered_doctor_information (Optional[dict], optional): Department-filtered doctor information
                                                                    to postprocess the schedule by tool_calling strategy.
            required_tests (Optional[list], optional): Required-test metadata used by the reasoning branch
                                                       to enrich each test_schedule entry.
            filtered_test_device_information (Optional[dict], optional): Test device information used by the
                                                                         reasoning branch to map device codes
                                                                         back to test codes.
            utc_offset (Optional[str], optional): UTC offset used by the reasoning branch to build ISO timestamps.
            rule (Optional[SchedulingRule], optional): `SchedulingRule` instance used by the reasoning branch to
                                               deterministically compute the follow-up consultation slot
                                               via `physician_filter` + `find_earliest_time`.
            attending_physician (Optional[str], optional): Name of the patient's attending physician; used by
                                                            the reasoning branch when computing `fu_schedule`.

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

                assert 'test_schedule' in text_dict
                assert isinstance(text_dict['test_schedule'], list)

                code_to_test = {t['test_code']: t for t in (required_tests or [])}
                device_to_code = {
                    dev: code
                    for code, info in (filtered_test_device_information or {}).get('test', {}).items()
                    for dev in info.get('devices', {})
                }

                latest, test_visit_dates = None, set()
                for entry in text_dict['test_schedule']:
                    assert isinstance(entry, dict) and len(entry) == 1
                    dev = next(iter(entry))
                    slot = entry[dev]
                    start = float(slot['start'])
                    end = float(slot['end'])
                    date = str(slot['date'])
                    code = device_to_code[dev]
                    t = code_to_test[code]
                    end_iso = get_iso_time(end, date, utc_offset)
                    result_ready_at = add_hours_to_iso(end_iso, t['result_hours'])
                    entry[dev] = {
                        'name': t['name'],
                        'code': code,
                        'device': dev,
                        'date': date,
                        'start': start,
                        'end': end,
                        'result_ready_at': result_ready_at,
                        'priority': t['priority'],
                    }
                    test_visit_dates.add(date)
                    if latest is None or compare_iso_time(result_ready_at, latest):
                        latest = result_ready_at
                text_dict['test_visit_dates'] = list(test_visit_dates)
                text_dict['all_results_ready_at'] = latest

                # Follow-up consultation slot — computed deterministically here, NOT taken from the LLM.
                text_dict['fu_schedule'] = None
                if (
                    rule is not None
                    and attending_physician
                    and filtered_doctor_information
                    and latest is not None
                ):
                    candidates = rule.physician_filter(
                        filtered_doctor_information,
                        preferred_doctor=attending_physician,
                        min_time=latest,
                    )
                    earliest = rule.find_earliest_time(candidates)
                    if earliest['doctor'] and earliest['schedule']:
                        doctor = earliest['doctor'][0]
                        iso = earliest['schedule'][0]
                        duration = filtered_doctor_information['doctor'][doctor]['outpatient_duration']
                        st_hour = iso_to_hour(iso)
                        end_hour = float(Decimal(str(st_hour)) + Decimal(str(duration)))
                        text_dict['fu_schedule'] = {
                            doctor: {
                                'date': iso_to_date(iso),
                                'start': st_hour,
                                'end': end_hour,
                            }
                        }
                return text_dict

            except:
                return str(data)
        
        elif strategy == 'tool_calling':
            schedule = {
                'test_schedule': [], 
                'test_visit_dates': data['test_visit_dates'], 
                'fu_schedule': None, 
                'all_results_ready_at': data['all_results_ready_at']
            }
            
            # Follow-up visit schedule post-processing
            fu_schedule = data['fu_schedule']
            if fu_schedule['doctor'] and fu_schedule['schedule']:
                doctor = fu_schedule['doctor'][0]
                duration = filtered_doctor_information['doctor'][doctor]['outpatient_duration']
                date, st_hour = iso_to_date(fu_schedule['schedule'][0]), iso_to_hour(fu_schedule['schedule'][0])
                tr_hour = float(Decimal(str(duration)) + Decimal(str(st_hour)))
                schedule['fu_schedule'] = {doctor: {'date': date, 'start': st_hour, 'end': tr_hour}}
            
            # Test post-processing
            required_tests = data['test_schedule']
            for _, values in required_tests.items():
                device_code = values['device']
                st_hour, tr_hour = iso_to_hour(values['start']), iso_to_hour(values['end'])
                tmp_schedule = {**values}
                tmp_schedule['start'] = st_hour
                tmp_schedule['end'] = tr_hour
                schedule['test_schedule'].append(
                    {device_code: tmp_schedule}
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
        attending_physician = known_condition['attending_physician']
        
        # First, try to use the tool calling
        try:
            assert self.scheduling_strategy == 'tool_calling', log('Scheduling strategy is set to `reasoning`, directly use the reasoning method.', level='warning')
            
            # Invoke
            prediction = scheduling_tool_calling(
                client=client, 
                user_prompt = known_condition['patient_intention'] + f' (Attending Physician: {attending_physician})' \
                    if attending_physician else known_condition['patient_intention'],
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
                        prediction['result'] = "Sorry, we couldn't find a matching information. Could you please check your details again (patient and doctor names)?"
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
                    filtered_doctor_information = self.environment.get_doctor_schedule(
                        doctor_information=doctor_information,
                        department=department,
                        fhir_integration=self.fhir_integration and doctor_information is None,
                    )
                    test_schedules = OPFUSchedulingSimulation.postprocessing(
                        strategy='tool_calling',
                        data=res,
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
        except Exception as e:
            log(f'Exception occured: {e}', 'warning')
            # Fallback only applies after retrieve_patient_tests has populated required_tests.
            if not known_condition.get('required_tests'):
                raise ToolCallingError(colorstr('red', 'Reasoning fallback without test retrieval.'))

            if self.scheduling_strategy == 'tool_calling':
                log('Failed to select an appropriate tool. Falling back to reasoning-based scheduling.', level='warning')

            required_test_codes = [t['test_code'] for t in known_condition['required_tests']]
            filtered_doctor_information = self.environment.get_doctor_schedule(
                doctor_information=doctor_information,
                department=department,
                fhir_integration=self.fhir_integration and doctor_information is None,
            )
            filtered_test_device_information = self.environment.get_test_device_schedule(
                test_device_information=test_device_information,
                test_code=required_test_codes,
                fhir_integration=self.fhir_integration and test_device_information is None,
            )
            current_time = f"{self.environment.current_time} (Date: {iso_to_date(self.environment.current_time)}, Time: {round(iso_to_hour(self.environment.current_time), 3)})"
            user_prompt = self.admin_staff_agent.scheduling_user_prompt_template.format(
                START_HOUR=self._START_HOUR,
                END_HOUR=self._END_HOUR,
                TIME_UNIT=self._TIME_UNIT,
                CURRENT_TIME=current_time,
                DEPARTMENT=department,
                PREFERENCE=known_condition['patient_intention'],
                DAY=self._DAY,
                TESTS=json.dumps(known_condition['required_tests'], indent=2),
                TEST_DEVICES=json.dumps(filtered_test_device_information, indent=2),
            )

            tries, schedule = 0, None
            while 1:
                schedule = self.admin_staff_agent(
                    user_prompt,
                    using_multi_turn=False,
                    verbose=False,
                    **kwargs,
                )
                schedule = OPFUSchedulingSimulation.postprocessing(
                    strategy='reasoning',
                    data=schedule,
                    filtered_doctor_information=filtered_doctor_information,
                    required_tests=known_condition['required_tests'],
                    filtered_test_device_information=filtered_test_device_information,
                    utc_offset=self.environment._utc_offset,
                    rule=self.rules,
                    attending_physician=known_condition.get('attending_physician'),
                )
                if isinstance(schedule, dict) or tries >= reasoning_max_tries:
                    break
                tries += 1

            if not isinstance(schedule, dict):
                self.admin_staff_agent.reset_history(verbose=False)
                raise SchedulingError(colorstr('red', 'Reasoning fallback failed to produce a valid schedule JSON.'))

            prediction = {
                'type': 'tool',
                'result': schedule,
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
            Tuple[dict, dict, dict, dict]: 
                - Doctor information.
                - Test information
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
                                pred_test_schedules = pred_schedule['test_schedule']

                                # Build a humanized summary of every test slot
                                parts = []
                                for entry in pred_test_schedules:
                                    for _, slot in entry.items():
                                        parts.append(
                                            f"{slot['name']} on {slot['date']} from {slot['start']} to {slot['end']}"
                                        )

                                # Notify the patient of any required tests that the agent could not
                                # fit within the simulation window (no deferred booking is attempted).
                                required_test_codes = {t['test_code'] for t in gt_patient_condition.get('required_tests', [])}
                                unscheduled_tests = {value['name'] for entry in pred_test_schedules for value in entry.values() if value['code'] not in required_test_codes}
                                if unscheduled_tests:
                                    parts.append(
                                        f"however, the scheduling for {', '.join(sorted(unscheduled_tests))} test(s) will be arranged later"
                                    )

                                fu_slot = pred_schedule['fu_schedule']
                                fu_doctor = staff_known_data['patient_fv']['attending_physician']
                                if isinstance(fu_slot, dict) and fu_slot:
                                    fu_info = fu_slot[fu_doctor]
                                    parts.append(
                                        f"follow-up with {fu_doctor} on {fu_info['date']} from {fu_info['start']} to {fu_info['end']}"
                                    )
                                else:
                                    all_results_ready_at = pred_schedule.get('all_results_ready_at')
                                    if all_results_ready_at:
                                        parts.append(
                                            f"and can I make an follow-up appointment with {fu_doctor} after {all_results_ready_at}"
                                        )
                                    else:
                                        parts.append(f"(follow-up with {fu_doctor} cannot be booked at this time)")
                                
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
                        return doctor_information, test_device_information, result_dict, token_usage

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
                        doctor_information=doctor_information,
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
            return doctor_information, test_device_information, result_dict, token_usage
        
        except ToolCallingError:
            result_dict = {
                'gt': [gt_patient_condition],
                'pred': [None],
                'status': [False],
                'status_code': [STATUS_CODES['test_retrieve']['identify']],
                'dialog': [preprocess_dialog(self.dialog_history['test_scheduling'])]
            }
            log("Simulation completed.", color=True)
            token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
            return doctor_information, test_device_information, result_dict, token_usage

        except SchedulingError:
            result_dict = {
                'gt': [gt_patient_condition],
                'pred': [None],
                'status': [False],
                'status_code': [STATUS_CODES['format']],
                'dialog': [preprocess_dialog(self.dialog_history['test_scheduling'])]
            }
            log("Simulation completed.", color=True)
            token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
            return doctor_information, test_device_information, result_dict, token_usage
        
        # Otherwise
        except Exception as e:
            status_code = STATUS_CODES['unexpected'].format(e=e)
            log(status_code, level='warning')
            result_dict = {
                'gt': [gt_patient_condition],
                'pred': [None],
                'status': [False],
                'status_code': [status_code],
                'dialog': [preprocess_dialog(self.dialog_history['test_scheduling'])]
            }
        
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
                fu_slot = pred_schedule['fu_schedule']
                fu_schedule = fu_slot[next(iter(fu_slot))] if fu_slot else None
                prediction = {
                    'visit_type': 'follow_up_visit',
                    'patient': staff_known_data['patient_fv']['patient'],
                    'attending_physician': staff_known_data['attending_physician'],
                    'department': staff_known_data['department'],
                    'date': fu_schedule['date'] if fu_schedule else None,
                    'schedule': [fu_schedule['start'], fu_schedule['end']] if fu_schedule else None,
                    'patient_intention': staff_known_data['patient_intention'],
                    'preference': gt_data[i].get('preference'),
                    'preferred_doctor': gt_data[i].get('preferred_doctor'),
                    'valid_from': gt_data[i].get('valid_from'),
                    'test': pred_schedule['test_schedule'],
                    'last_updated_time': self.environment.current_time
                }
                result_dict['pred'] = [prediction]
                result_dict['status'] = [True]
            except Exception:
                result_dict['status_code'] = [STATUS_CODES['format']]
                log('Error while organizing the prediction. Returning a failure result.', level='warning')

        log("Simulation completed.", color=True)
        token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}

        return doctor_information, test_device_information, result_dict, token_usage
