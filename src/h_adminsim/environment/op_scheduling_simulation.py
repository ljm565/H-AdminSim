import re
import os
import json
import random
from importlib import resources
from patientsim import PatientAgent
from decimal import Decimal, getcontext
from typing import Tuple, Union, Optional
from langchain_core.messages import HumanMessage, AIMessage

from h_adminsim import AdminStaffAgent
from h_adminsim.registry import PREFERENCE_PHRASE_PATIENT, PREFERENCE_PHRASE_STAFF
from h_adminsim.environment.hospital import HospitalEnvironment
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.common_utils import iso_to_hour, iso_to_date
from h_adminsim.tools import SchedulingRule, scheduling_tool_calling



class OPScehdulingSimulation:
    def __init__(self,
                 patient_agent: PatientAgent,
                 admin_staff_agent: AdminStaffAgent,
                 metadata: dict,
                 environment: HospitalEnvironment,
                 preference_rejection_prob: float = 0.3,
                 preferene_rejection_prob_decay: float = 0.5,
                 fhir_integration: bool = False,
                 schedule_rejection_prompt_path: Optional[str] = None):
        
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
        self.rules = SchedulingRule(metadata, self.environment)
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
        self.patient_agent.client.reset_history(verbose=verbose)
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

    
    def postprocessing(self, 
                       strategy: str, 
                       text: Union[str, dict],
                       filtered_doctor_information: Optional[dict] = None) -> Union[str, dict]:
        """
        Attempts to parse the given text as JSON. If parsing succeeds, returns a dictionary;
        otherwise, returns the original string.

        Args:
            strategy (str): Scheduling strategy. It must be either `llm` or `tool_calling`.
            text (Union[str, dict]): The text output to post-process, potentially a JSON-formatted string. 
            filtered_doctor_information (Optional[dict], optional): Department-filtered doctor information 
                                                                    to postprocess the schedule by tool_calling strategy.

        Returns:
            Union[str, dict]: A dictionary if the text is valid JSON, otherwise the original string.
        """
        if strategy == 'llm':
            try:
                if isinstance(text, str):
                    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                        text_dict = json.loads(json_str)
                    else:
                        try:
                            text_dict = json.loads(text)
                        except:
                            return text
                else:
                    text_dict = text
                
                assert len(text_dict) == 1 and all(k in text_dict for k in ['schedule'])   # Basic sanity check
                key = list(text_dict['schedule'].keys())[0]
                text_dict['schedule'][key]['start'] = float(text_dict['schedule'][key]['start'])
                text_dict['schedule'][key]['end'] = float(text_dict['schedule'][key]['end'])
                text_dict['schedule'][key]['date'] = str(text_dict['schedule'][key]['date'])
                return text_dict
            
            except:
                return str(text)
        
        elif strategy == 'tool_calling':
            doctor = text['doctor'][0]
            duration = filtered_doctor_information['doctor'][doctor]['outpatient_duration']
            date, st_hour = iso_to_date(text['schedule'][0]), iso_to_hour(text['schedule'][0])
            tr_hour = float(Decimal(str(duration)) + Decimal(str(st_hour)))
            return {'schedule': {doctor: {'date': date, 'start': st_hour, 'end': tr_hour}}}



    def scheduling(self,
                   scheduling_strategy: str,
                   known_condition: dict,
                   doctor_information: dict, 
                   environment: HospitalEnvironment, 
                   reschedule_flag: bool = False, 
                   **kwargs) -> Tuple[bool, str, Union[str, dict]]:
        """
        Make an appointment between the doctor and the patient.

        Args:
            scheduling_strategy (str): Scheduling strategy.
            known_condition (dict): Patient conditions known to the staff.
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                       including availability and other relevant details.
            environment (HospitalEnvironment): Hospital environment.
            reschedule_flag (bool, optional): Whether this process is rescheduling or not. Defaults to False.

        Return
            Tuple[bool, str, Union[str, dict], dict, list[str]]: 
                - A boolean indicating whether the prediction passed all sanity checks.
                - A string explaining its status.
                - The original prediction (wrong case) or processed prediction (correct case) (either unchanged or used for debugging/logging if invalid).
        """
        department = known_condition['department']
        reschedule_desc = "Rescheduling requested. This is the rescheduling of a patient who wishes to move their appointment earlier due to a previous patient's cancelled reservation" \
            if reschedule_flag else 'Not requested.'
        
        if reschedule_flag:
            log(f'\n{colorstr("RESCEDULE")}: Rescheduling occured.')
        
        ################################# LLM-based Scheduling #################################
        if scheduling_strategy == 'llm':
            filtered_doctor_information = environment.get_doctor_schedule(
                doctor_information=doctor_information if not self.fhir_integration else None,
                department=department,
                fhir_integration=self.fhir_integration,
                express_detail=True
            )
            user_prompt = self.admin_staff_agent.scheduling_user_prompt_template.format(
                START_HOUR=self._START_HOUR,
                END_HOUR=self._END_HOUR,
                TIME_UNIT=self._TIME_UNIT,
                CURRENT_TIME=environment.current_time,
                DEPARTMENT=department,
                PREFERENCE=known_condition['patient_intention'],
                RESCHEDULING_FLAG=reschedule_desc,
                DAY=self._DAY,
                DOCTOR=json.dumps(filtered_doctor_information, indent=2),
            )
            prediction = self.admin_staff_agent(
                user_prompt,
                using_multi_turn=False,
                verbose=False,
                **kwargs,
            )
            prediction = self.postprocessing(
                scheduling_strategy,
                prediction,
            )
            self.admin_staff_agent.reset_history(verbose=False)
        
        ############################# Tool calling-based Scheduling ############################
        elif scheduling_strategy == 'tool_calling':
            filtered_doctor_information = environment.get_doctor_schedule(
                doctor_information=doctor_information if not self.fhir_integration else None,
                department=department,
                fhir_integration=self.fhir_integration,
            )
            self.client = self.admin_staff_agent.build_agent(
                rule=self.rules, 
                doctor_info=filtered_doctor_information
            )
            try:
                prediction = scheduling_tool_calling(
                    client=self.client, 
                    user_prompt=known_condition['patient_intention']
                )['result']
                prediction = self.postprocessing(
                    scheduling_strategy,
                    prediction,
                    filtered_doctor_information,
                )
            except:
                log('Fail to load an appropriate tool', level='warning')
                prediction = 'Fail to load an appropriate tool'

        else:
            raise NotImplementedError(
                colorstr('red', 'Unsupported strategy. Supported strategies are ["llm", "tool_calling"].')
            )

        return prediction
    

    def canceling(self, patient_intention: str) -> Union[int, str]:
        """
        Handle a multi-turn appointment cancellation request using a tool-calling agent.

        Args:
            patient_intention (str): The patient's utterance expressing a cancellation request.

        Returns:
            Union[int, str]: The cancellation result or a clarification message to the patient.
        """
        chat_history = self._to_lc_history('cancel')
        prediction = scheduling_tool_calling(
            client=self.client,
            user_prompt=patient_intention,
            history=chat_history,
        )

        if prediction['type'] == 'tool':
            if prediction['result'] == -1:
                return "Sorry, we couldn't find a matching appointment. Could you please check your appointment details again?"
            else:
                return prediction['result']
 
        return prediction['result']
    

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


    def scheduling_simulate(self,
                            strategy: str,
                            gt_data: dict,
                            staff_known_data: dict,
                            doctor_information: dict,
                            verbose: bool = False,
                            patient_kwargs: dict = {},
                            staff_kwargs: dict = {},
                            **kwargs):
        """
        Simulate a multi-turn outpatient scheduling dialogue between a patient agent and an administrative staff agent.

        Args:
            strategy (str): Scheduling strategy used by the staff agent.
            gt_data (dict): Ground-truth patient condition(s) for each dialogue turn.
            staff_known_data (dict): Patient information known to the staff agent at each turn.
            doctor_information (dict): Available doctor and schedule information.
            verbose (bool, optional): Whether to log detailed simulation outputs. Defaults to False.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff scheduling function.
            **kwargs: Shared keyword arguments passed to both agents.

        Yields:
            dict: Scheduling proposal generated by the staff agent at each turn.
        """
        # Initialize agents
        self._init_agents(verbose=verbose)
        
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
            pred_schedule = self.scheduling(
                strategy,
                staff_known_condition,
                doctor_information,
                self.environment,
                **staff_kwargs
            )
            staff_response = self.admin_staff_agent.staff_suggestion.format(schedule=pred_schedule)
            self.dialog_history['scheduling'].append({"role": "Staff", "content": staff_response})
            role = f"{colorstr('blue', 'Staff')}"
            log(f"{role:<25}: {staff_response}")
            
            yield pred_schedule

            # Preference rejection logic
            ## Rejection case
            if random.random() < preference_reject_prob and i != len(gt_data) - 1:
                preference_reject_prob *= self.preferene_rejection_prob_decay
            ## Non-rejection case
            else:
                self.dialog_history['scheduling'].append({"role": "Patient", "content": self.end_phrase})
                role = f"{colorstr('green', 'Patient')} ({gt_data[i]['preference']})"
                log(f"{role:<25}: {self.end_phrase}")
                log("Simulation completed.", color=True)
                break

    
    def canceling_simulate(self, 
                           patient_schedules: list[dict],
                           verbose: bool,
                           max_inferences: int = 5) -> int:
        # Initialize agents
        self._init_agents(verbose=verbose)
        self.client = self.admin_staff_agent.build_agent(
            rule=self.rules, 
            patient_schedule_list=patient_schedules
        )

        # Start conversation
        staff_greet = self.admin_staff_agent.general_staff_greet
        self.dialog_history['cancel'].append({"role": "Staff", "content": staff_greet})
        role = f"{colorstr('blue', 'Staff')}"
        log(f"{role:<25}: {staff_greet}")

        for _ in range(max_inferences):
            # Obtain response from patient
            patient_response = self.patient_agent(
                self.dialog_history['cancel'][-1]["content"],
                using_multi_turn=True,
                verbose=False,
            )
            self.dialog_history['cancel'].append({"role": "Patient", "content": patient_response})
            role = f"{colorstr('green', 'Patient')} ({('cancel')})"
            log(f"{role:<25}: {patient_response}")

            # Canceling from staff
            staff_response = self.canceling(
                patient_intention=patient_response,
            )
            if isinstance(staff_response, str):
                break_flag = False
            else:
                break_flag = True
                cancelled_schedule_index = staff_response
                cancelled_schedule = {k: v for k, v in patient_schedules[staff_response].items() \
                                      if k in ['patient', 'attending_physician', 'department', 'date', 'schedule']}
                staff_response = f"I will cancel this schedule: {cancelled_schedule}"
            
            self.dialog_history['cancel'].append({"role": "Staff", "content": staff_response})
            role = f"{colorstr('blue', 'Staff')}"
            log(f"{role:<25}: {staff_response}")

            if break_flag:
                self.dialog_history['cancel'].append({"role": "Patient", "content": self.end_phrase})
                role = f"{colorstr('blue', 'Patient')}"
                log(f"{role:<25}: {self.end_phrase}")
                return cancelled_schedule_index
    
        return -1
