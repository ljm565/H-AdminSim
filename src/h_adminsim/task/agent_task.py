import os
import json
import random
from copy import deepcopy
from importlib import resources
from decimal import Decimal, getcontext
from typing import Tuple, Union, Optional
from dotenv import load_dotenv, find_dotenv

from patientsim import PatientAgent
from patientsim import AdminStaffAgent as IntakeAdminStaffAgent
from patientsim.environment import OPSimulation as OPFVIntakeSimulation

from h_adminsim import SupervisorAgent
from h_adminsim import AdminStaffAgent as SchedulingAdminStaffAgent
from h_adminsim.environment.hospital import HospitalEnvironment
from h_adminsim.environment import OPScehdulingSimulation as OPFVScheduleSimulation
from h_adminsim.tools import DataConverter, SchedulingRule
from h_adminsim.registry.errors import ScheduleNotFoundError
from h_adminsim.registry import STATUS_CODES, PREFERENCE_PHRASE_PATIENT
from h_adminsim.utils import colorstr, log
from h_adminsim.utils.fhir_utils import *
from h_adminsim.utils.common_utils import *



class FirstVisitOutpatientTask:
    def __init__(self):
        self.token_stats = {
            'patient_token': {'input':[], 'output': [], 'reasoning': []}, 
            'admin_staff_token': {'input': [], 'output': [], 'reasoning': []}, 
            'supervisor_token': {'input':[], 'output': [], 'reasoning': []}
        }

    
    def save_token_data(self, 
                        patient_token: Optional[dict] = None, 
                        admin_staff_token: Optional[dict] = None, 
                        supervisor_token: Optional[dict] = None):
        """
        Save the API token usage data

        Args:
            patient_token (Optional[dict], optional): Patient token information. Defaults to None.
            admin_staff_token (Optional[dict], optional): Administration staff token information. Defaults to None.
            supervisor_token (Optional[dict], optional): Supervisor token information. Defaults to None.
        """
        if patient_token:
            self.token_stats['patient_token']['input'].extend(patient_token['prompt_tokens'])
            self.token_stats['patient_token']['output'].extend(patient_token['completion_tokens'])
            if 'reasoning_tokens' in patient_token:
                self.token_stats['patient_token']['reasoning'].extend(patient_token['reasoning_tokens'])

        if admin_staff_token:
            self.token_stats['admin_staff_token']['input'].extend(admin_staff_token['prompt_tokens'])
            self.token_stats['admin_staff_token']['output'].extend(admin_staff_token['completion_tokens'])
            if 'reasoning_tokens' in admin_staff_token:
                self.token_stats['admin_staff_token']['reasoning'].extend(admin_staff_token['reasoning_tokens'])

        if supervisor_token:
            self.token_stats['supervisor_token']['input'].extend(supervisor_token['prompt_tokens'])
            self.token_stats['supervisor_token']['output'].extend(supervisor_token['completion_tokens'])
            if 'reasoning_tokens' in supervisor_token:
                self.token_stats['supervisor_token']['reasoning'].extend(supervisor_token['reasoning_tokens'])
    
    
    def _init_task_models(self, model: str, vllm_endpoint: Optional[str] = None) -> Tuple[str, str, bool]:
        """
        Initialize the model for the task.

        Args:
            model (str): The model name.
            vllm_endpoint (Optional[str], optional): The VLLM endpoint URL. Defaults to None.
        
        Returns:
            Tuple[str, str, bool]: The model name, VLLM endpoint URL, vllm usage flag.
        """
        if any(keyword in model.lower() for keyword in ['gemini', 'gpt']):
            return model, None, False
        else:
            assert vllm_endpoint is not None, log('VLLM endpoint must be provided for non-Gemini/GPT models.', 'error')
            return model, vllm_endpoint, True



class OutpatientFirstIntake(FirstVisitOutpatientTask):
    def __init__(self, 
                 patient_model: str,
                 admin_staff_model: str,
                 supervisor_agent: Optional[SupervisorAgent] = None,
                 intake_max_inference: int = 5,
                 max_retries: int = 8,
                 admin_staff_last_task_user_prompt_path: Optional[str] = None,
                 patient_vllm_endpoint: Optional[str] = None,
                 admin_staff_vllm_endpoint: Optional[str] = None):
        super().__init__()
        
        # Initialize variables
        self.name = 'intake'
        self.patient_model, self.patient_vllm_endpoint, self.patient_use_vllm \
            = self._init_task_models(patient_model, patient_vllm_endpoint)
        self.admin_staff_model, self.admin_staff_vllm_endpoint, self.admin_staff_use_vllm \
            = self._init_task_models(admin_staff_model, admin_staff_vllm_endpoint)
        self.use_supervisor = True if isinstance(supervisor_agent, SupervisorAgent) else False
        self.supervisor_client = supervisor_agent if self.use_supervisor else None
        task_mechanism = 'Staff + Supervisor' if self.use_supervisor else 'Staff'
        self.max_inferences = intake_max_inference
        self.max_retries = max_retries
        self._init_last_task_prompt(admin_staff_last_task_user_prompt_path)
        self.patient_reasoning_kwargs = {'reasoning_effort': 'low'} if 'gpt-5' in self.patient_model.lower() else {}
        self.staff_reasoning_kwargs = {'reasoning_effort': 'low'} if 'gpt-5' in self.admin_staff_model.lower() else {}
        log(f'Patient intake tasks are conducted by {colorstr(task_mechanism)}')
    

    def _init_last_task_prompt(self, admin_staff_last_task_user_prompt_path: Optional[str] = None) -> str:
        """
        Initialize the user prompt for the admnistration staff agent's last task.

        Args:
            admin_staff_last_task_user_prompt_path (Optional[str], optional): Path to a custom user prompt file. 
                                                                              If not provided, the default user prompt will be used. Defaults to None.

        Returns:
            str: The user prompt.

        Raises:
            FileNotFoundError: If the specified user prompt file does not exist.
        """
        if not self.use_supervisor:
            if not admin_staff_last_task_user_prompt_path:
                prompt_file_name = 'intake_staff_task_user.txt'
                file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
                self.last_task_user_prompt = file_path.read_text()
            else:
                if not os.path.exists(admin_staff_last_task_user_prompt_path):
                    raise FileNotFoundError(colorstr("red", f"User prompt file not found: {admin_staff_last_task_user_prompt_path}"))
                with open(admin_staff_last_task_user_prompt_path, 'r') as f:
                    self.last_task_user_prompt = f.read()
        else:
            if admin_staff_last_task_user_prompt_path:
                log('The admin_staff_last_task_user_prompt_path setting is ignored when using supervisor model.', 'warning')
    
    
    @staticmethod
    def postprocessing_department(text: str) -> str:
        """
        Post-processing method of text output, especially for the department decision.

        Args:
            text (str): Text input.

        Returns:
            str: Post-processed text output.
        """
        try:
            pattern = re.compile(r'Answer:\s*\d+\.\s*(.+)')
            text = pattern.search(text).group(1)
        except:
            text = 'wrong'
        return text

    
    @staticmethod
    def postprocessing_information(text: str) -> Union[str, dict]:
        """
        Post-processing method of text output, especially for the patient information extraction.

        Args:
            text (str): Text input.

        Returns:
            Union[str, dict]: A dictionary if the text is valid JSON, otherwise the original string.
        """
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
            
            assert len(text_dict) == 6 and all(k in text_dict for k in ['name', 'gender', 'phone_number', 'personal_id', 'address', 'department'])   # Basic sanity check
            return text_dict
        except:
            return str(text)
        
    
    def _department_decision(self, prediction_department: str, prediction_supervison: Union[str, dict], gt_department: str) -> Tuple[str, list[str]]:
        """
        Determine the final department decision by considering both 
        the interaction agent result and the supervisor agent result.

        Args:
            prediction_department (str): The department predicted by the interaction agent.
            prediction_supervison (Union[str, dict]): The supervisor agent's result. 
                If this is a dictionary, it should contain a 'department' field.
            gt_department (str): The ground truth department for the patient.

        Returns:
            str: The final department decision.
        """
        try:
            sup_department = prediction_supervison.pop('department')
            if prediction_department == sup_department:
                trial = ['match']
            else:
                if prediction_department in gt_department and sup_department not in gt_department:
                    trial = ['mismatch - worse']
                elif prediction_department not in gt_department and sup_department in gt_department:
                    trial = ['mismatch - better']
                else:
                    trial = ['mismatch - both wrong']
            return sup_department, trial
        except:
            return prediction_department, ['supervisor error']
        
    
    def _sanity_check(self,
                      prediction: dict, 
                      gt: dict,
                      conversations: str) -> Tuple[bool, str]:
        """
        Performs a sanity check on the predicted patient information and department against the ground truth.

        Args:
            prediction (dict): The output generated by the model. Expected to contain:
                - 'patient': dict of patient demographic information (e.g., name, birth date, gender)
                - 'department': str representing the predicted department
            gt (dict): The ground truth data. Expected to contain:
                - 'patient': dict of correct patient demographic information
                - 'department': str of the correct department
            conversations (str): The full conversation text between the patient and administration staff.

        Returns:
            Tuple[bool, str]:
                - bool: True if the prediction passes all sanity checks, False otherwise
                - str: Status code indicating the type of check passed or failed
        """
        ############################ Check the prediciton format #############################
        if not isinstance(prediction['patient'], dict):
            return False, STATUS_CODES['format']  # Could not be parsed as a dictionary
        
        ############################ Incomplete simulation case #############################
        if not all(v.lower() in conversations.lower() for k, v in gt['patient'].items()):
            return False, STATUS_CODES['simulation']
        
        ############################ Check with the ground truth #############################
        wrong_department = prediction['department'][0] not in gt['department']
        wrong_info = prediction['patient'] != gt['patient']
        if wrong_department and wrong_info:
            return False, STATUS_CODES['department & patient']
        elif wrong_department:
            return False, STATUS_CODES['department']
        elif wrong_info:
            return False, STATUS_CODES['patient']
        
        return True, STATUS_CODES['correct']
        

    def __call__(self, data_pair: Tuple[dict, dict], agent_test_data: dict, agent_results: dict, environment, verbose: bool = False) -> dict:
        """
        Estimates the most appropriate medical department for each patient using an LLM agent.

        Args:
            data_pair (Tuple[dict, dict]): A pair of ground truth and patient data for agent simulation.
            agent_test_data (dict): A dictionary containing test data for a single hospital.
                Expected to include:
                    - 'department': Dictionary of available departments.
            agent_results (dict): Placeholder for compatibility; not used in this method.
            environment (HospitalEnvironment): Hospital environment instance to manage patient schedules.
            verbose (bool): Whether logging the each result or not.

        Returns:
            dict: A dictionary with:
                - 'gt': List of ground-truth departments.
                - 'pred': List of predicted departments from the LLM agent.
                - 'status': List of booleans indicating whether each prediction correct.
                - 'status_code': List of status codes explaining each status.
        """
        gt, test_data = data_pair
        departments = list(agent_test_data['department'].keys())
        results = init_result_dict()
        
        # Append a ground truth
        name, gender, birth_date, telecom, personal_id, address = \
            gt['patient'], gt['gender'], gt['birthDate'], gt['telecom'][0]['value'], gt['identifier'][0]['value'], gt['address'][0]['text']
        gt_data = {
            'patient': {
                'name': name,
                'gender': gender,
                'phone_number': telecom,
                'personal_id': personal_id,
                'address': address,
            }, 
            'department': gt['department']
        }
        results['gt'].append(gt_data)

        # LLM call: Conversation and department decision
        department_candidates = test_data['constraint']['symptom']['department']
        if test_data['constraint']['symptom_level'] == 'simple':
            medical_history = "None. This is the patient's first visit."
            diagnosis = "Unknown for now, as this is the patient's first visit to the hospital."
        elif test_data['constraint']['symptom_level'] == 'with_history':
            medical_history = f"Diagnosed with {test_data['constraint']['symptom']['disease']} at a primary or secondary hospital."
            diagnosis = test_data['constraint']['symptom']['disease']
        else:
            log("Patient's symptom level must be either 'simple' or 'with_history'.", "error")
        
        # Simulation patient intake
        patient_agent = PatientAgent(
            self.patient_model,
            'outpatient',
            lang_proficiency_level='B',
            recall_level='no_history' if test_data['constraint']['symptom_level'] == 'simple' else 'high',
            use_vllm=self.patient_use_vllm,
            vllm_endpoint=self.patient_vllm_endpoint,
            department=department_candidates,
            name=name,
            birth_date=birth_date,
            gender=gender,
            telecom=telecom,
            personal_id=personal_id,
            address=address,
            medical_history=medical_history,
            diagnosis=diagnosis,
            chiefcomplaint=test_data['constraint']['symptom']['symptom'],
            temperature=0 if not 'gpt-5' in self.patient_model.lower() else 1
        )
        admin_staff_agent = IntakeAdminStaffAgent(
            self.admin_staff_model,
            departments,
            max_inferences=self.max_inferences,
            use_vllm=self.admin_staff_use_vllm,
            vllm_endpoint=self.admin_staff_vllm_endpoint,
            temperature=0 if not 'gpt-5' in self.admin_staff_model.lower() else 1
        )
        sim_environment = OPFVIntakeSimulation(patient_agent, admin_staff_agent, max_inferences=self.max_inferences)
        output = run_with_retry(
            sim_environment.simulate,
            verbose=False,
            patient_kwargs=self.patient_reasoning_kwargs, 
            staff_kwargs=self.staff_reasoning_kwargs,
            max_retries=self.max_retries,
        )
        dialogs, patient_token, admin_staff_token = output['dialog_history'], output.get('patient_token_usage'), output.get('admin_staff_token_usage')
        prediction_department = OutpatientFirstIntake.postprocessing_department(dialogs[-1]['content'])

        # LLM call: Agent which should extract demographic information of the patient and evaluation the department decision result
        dialogs = preprocess_dialog(dialogs)
        
        if self.use_supervisor:
            user_prompt = self.supervisor_client.user_prompt_template.format(
                CONVERSATION=dialogs,
                DEPARTMENTS=''.join([f'{i+1}. {department}\n' for i, department in enumerate(departments)])
            )
            prediction_supervision = run_with_retry(
                self.supervisor_client,
                user_prompt,
                using_multi_turn=False,
                verbose=False,
                max_retries=self.max_retries,
            )
        else:
            prediction_supervision = run_with_retry(
                admin_staff_agent,
                self.last_task_user_prompt,
                verbose=False,
                max_retries=self.max_retries,
                **self.staff_reasoning_kwargs,
            )

        prediction_supervision = OutpatientFirstIntake.postprocessing_information(prediction_supervision)
        
        # Append token data
        self.save_token_data(
            patient_token, 
            admin_staff_token, 
            supervisor_token=self.supervisor_client.client.token_usages if self.use_supervisor else {}
        )

        # Sanity check
        department, trial = self._department_decision(prediction_department, prediction_supervision, gt['department'])
        prediction = {'patient': prediction_supervision, 'department': [department]}
        status, status_code = self._sanity_check(
            prediction=prediction,
            gt=gt_data,
            conversations=dialogs
        )

        if verbose:
            log(f'GT    : {gt_data}')
            log(f'Pred  : {prediction}')
            log(f'Status: {status_code}\n\n\n')
        
        # Append results
        results['pred'].append(prediction)
        results['status'].append(status)
        results['status_code'].append(status_code)
        results['trial'].append(trial)
        results['dialog'].append(dialogs)

        return results



class OutpatientFirstScheduling(FirstVisitOutpatientTask):
    def __init__(self, 
                 patient_model: str,
                 admin_staff_model: str,
                 schedule_cancellation_prob: float = 0.05,
                 request_early_schedule_prob: float = 0.1,
                 preference_rejection_prob: float = 0.3,
                 preferene_rejection_prob_decay: float = 0.5,
                 fhir_integration: bool = False,
                 scheduling_max_inference: int = 5,
                 max_retries: int = 8,
                 patient_vllm_endpoint: Optional[str] = None,
                 admin_staff_vllm_endpoint: Optional[str] = None):
        super().__init__()

        # Initialize variables
        getcontext().prec = 10
        dotenv_path = find_dotenv(usecwd=True)
        load_dotenv(dotenv_path, override=True)
        self.name = 'schedule'
        self.patient_model, self.patient_vllm_endpoint, self.patient_use_vllm \
            = self._init_task_models(patient_model, patient_vllm_endpoint)
        self.admin_staff_model, self.admin_staff_vllm_endpoint, self.admin_staff_use_vllm \
            = self._init_task_models(admin_staff_model, admin_staff_vllm_endpoint)
        
        # Initialize scheduling methods and a staff agent
        self.admin_staff_agent = SchedulingAdminStaffAgent(
            target_task='first_outpatient_scheduling',
            model=self.admin_staff_model,
            use_vllm=self.admin_staff_use_vllm,
            vllm_endpoint=self.admin_staff_vllm_endpoint,
            temperature=0 if not 'gpt-5' in self.admin_staff_model.lower() else 1
        )

        # Scheduling parameters
        self.schedule_cancellation_prob = schedule_cancellation_prob
        self.request_early_schedule_prob = request_early_schedule_prob
        self.preference_rejection_prob = preference_rejection_prob
        self.preferene_rejection_prob_decay = preferene_rejection_prob_decay

        # Others
        self.fhir_integration = fhir_integration
        self.max_retries = max_retries
        self.max_inferences = scheduling_max_inference
        self.schedule_patient_system_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath('schedule_patient_system.txt'))
        self.cancel_patient_system_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath('cancel_patient_system.txt'))
        self.reschedule_patient_system_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath('reschedule_patient_system.txt'))
        self.patient_reasoning_kwargs = {'reasoning_effort': 'low'} if 'gpt-5' in self.patient_model.lower() else {}
        self.staff_reasoning_kwargs = {'reasoning_effort': 'low'} if 'gpt-5' in self.admin_staff_model.lower() else {}

    
    def _init_simulation(self,
                         system_prompt_path: str,
                         environment: HospitalEnvironment,
                         additional_patient_conditions: dict = {}) -> OPFVScheduleSimulation:
        """
        Initialize an outpatient first-visit intake and scheduling simulation.

        Args:
            system_prompt_path (str): Path to the system prompt used to initialize the patient agent.
            environment (HospitalEnvironment): Hospital environment configuration for the simulation.
            additional_patient_conditions (dict, optional): Additional patient-specific conditions for simulation control.

        Returns:
            OPFVIntakeSimulation: Configured outpatient intake and scheduling simulation instance.
        """
        patient_agent = PatientAgent(
            self.patient_model,
            'outpatient',
            use_vllm=self.patient_use_vllm,
            vllm_endpoint=self.patient_vllm_endpoint,
            system_prompt_path=system_prompt_path,
            log_verbose=False,
            additional_patient_conditions=additional_patient_conditions,
            temperature=0 if not 'gpt-5' in self.patient_model.lower() else 1
        )
        sim_environment = OPFVScheduleSimulation(
            patient_agent=patient_agent, 
            admin_staff_agent=self.admin_staff_agent, 
            metadata=self._metadata,
            department_data=self._department_data,
            environment=environment,
            preference_rejection_prob=self.preference_rejection_prob,
            preferene_rejection_prob_decay=self.preferene_rejection_prob_decay,
            fhir_integration=self.fhir_integration,
        )
        return sim_environment


    def get_intake_information(self, gt: dict, agent_results: dict, doctor_information: dict) -> Tuple[dict, str, bool]:
        """
        Extracts the patient name and predicted department from agent results.
        If predictions are not available, falls back to using ground truth labels.

        Args:
            gt (dict): Ground truth data of a patient.
            agent_results (dict): A dictionary that may contain predicted department results under the key 'department'.
            doctor_information (dict): Dictionary of doctor data including their existing schedules.
                                       Each key is a doctor's name, and each value includes a 'schedule' field.

        Returns:
            Tuple[str, str, bool]: Patient information, determined department, either predicted or ground truth and its sanity status.
        """
        # Prediction results are existing case
        try:
            for i, intake_gt in enumerate(agent_results['intake']['gt']):
                if gt['patient'] == intake_gt['patient']['name']:
                    break

            patient_info = agent_results['intake']['pred'][i]['patient']
            department = agent_results['intake']['pred'][i]['department'][0]
            sanity = agent_results['intake']['status'][i]

            assert gt['patient'] == agent_results['intake']['gt'][i]['patient']['name']
        
        # Loading from the ground truth
        except:
            log('The predicted department is not given. The ground truth value will be used.', 'warning')
            patient_info = {
                'name': gt['patient'],
                'gender': gt['gender'],
                'phone_number': gt['telecom'][0]['value'],
                'personal_id': gt['identifier'][0]['value'],
                'address': gt['address'][0]['text'],
            }
            department = doctor_information[gt['attending_physician']]['department']
            assert department in gt['department']
            sanity = True

        return patient_info, department, sanity
    

    def __check_is_earliest(self, 
                            prediction: dict, 
                            gt_patient_condition: dict, 
                            doctor_information: dict, 
                            environment: HospitalEnvironment) -> bool:
        """
        Check if the predicted schedule is the earliest possible option.

        Args:
            prediction (dict): Predicted schedule information including doctor, start time, end time, and date.
            gt_patient_condition (dict): Ground truth patient conditions used only for sanity checks.
            doctor_information (dict): Dictionary containing doctors' schedules and availability.
            environment (HospitalEnvironment): Environment object containing current time and UTC offset.

        Returns:
            bool: True if the predicted schedule is the earliest available, False otherwise.
        """
        # Init grount thruth values
        department = gt_patient_condition['department']
        preference_type = gt_patient_condition['preference']
        valid_from = gt_patient_condition['valid_from']
        fixed_schedules = environment.get_doctor_schedule(doctor_information=doctor_information, department=department)['doctor']

        # Get predicted results
        pred_doctor_name = list(prediction['schedule'].keys())[0]
        pred_start = prediction['schedule'][pred_doctor_name]['start']
        pred_end = prediction['schedule'][pred_doctor_name]['end']
        pred_date = prediction['schedule'][pred_doctor_name]['date']
        current_time = environment.current_time
        utc_offset = environment._utc_offset

        # Time segments
        prediction_schedule_segments = convert_time_to_segment(self._START_HOUR,
                                                               self._END_HOUR,
                                                               self._TIME_UNIT,
                                                               [pred_start, pred_end])
        
        for k, v in fixed_schedules.items():
            if preference_type == 'doctor' and k != pred_doctor_name:
                continue
            
            min_time_slot_n = int(Decimal(str(v['outpatient_duration'])) / Decimal(str(self._TIME_UNIT)))
            fixed_schedule = v['schedule']
            for date, schedule in fixed_schedule.items():
                # date > pred_date case
                if compare_iso_time(date, pred_date):
                    continue
                
                # valid_from > date case (preference == 'date' case)
                if valid_from and compare_iso_time(valid_from, date):
                    continue

                fixed_schedule_segments = sum([convert_time_to_segment(self._START_HOUR, 
                                                                       self._END_HOUR, 
                                                                       self._TIME_UNIT, 
                                                                       fs) for fs in schedule], [])
                all_time_segments = convert_time_to_segment(self._START_HOUR, self._END_HOUR, self._TIME_UNIT)
                free_time = [s for s in range(len(all_time_segments)) if s not in fixed_schedule_segments]
                
                if len(free_time):
                    valid_time_segments = [seg for seg in group_consecutive_segments(free_time) if len(seg) >= min_time_slot_n]
                    for valid_time in valid_time_segments:
                        if (valid_time[0] < prediction_schedule_segments[0] and pred_date == date) or (len(valid_time) and compare_iso_time(pred_date, date)):
                            free_max_st, _ = convert_segment_to_time(self._START_HOUR, self._END_HOUR, self._TIME_UNIT, [valid_time[0]])
                            free_max_st_iso = get_iso_time(free_max_st, date, utc_offset=utc_offset)
                            if compare_iso_time(free_max_st_iso, current_time):
                                return False
        return True
    

    def _sanity_check(self,
                      prediction: Union[str, dict], 
                      gt_patient_condition: dict,
                      doctor_information: dict,
                      environment: HospitalEnvironment) -> Tuple[bool, str]:
        """
        Validates a predicted schedule for a doctor by checking its structure, time validity, 
        duplication with existing schedules, and updates the doctor's schedule if valid.

        Args:
            prediction (Union[str, dict]): The predicted allocation result, either a string (if parsing failed)
                                           or a dictionary mapping a doctor's name to a schedule with 'start' and 'end' times.
            gt_patient_condition (dict): Ground truth patient conditions used only for sanity checks.
            doctor_information (dict): Dictionary of doctor data including their existing schedules.
                                       Each key is a doctor's name, and each value includes a 'schedule' field.
            environment (HospitalEnvironment): Hospital environment instance to manage patient schedules.

        Returns:
            Tuple[bool, str]: 
                - A boolean indicating whether the prediction passed all sanity checks.
                - A string explaining its status.
        """
        ############################ Check the prediciton format #############################
        if not isinstance(prediction, dict):
            return False, STATUS_CODES['format']    # Could not be parsed as a dictionary
        elif len(prediction['schedule']) > 1:
            return False, STATUS_CODES['conflict']['physician']    # Allocated more than one doctor; cannot determine target
    
        ################## Check the predicted schedule type and validities ##################
        try:
            pred_doctor_name = list(prediction['schedule'].keys())[0]
            start = prediction['schedule'][pred_doctor_name]['start']
            end = prediction['schedule'][pred_doctor_name]['end']
            date = prediction['schedule'][pred_doctor_name]['date']
            fixed_schedules = doctor_information[pred_doctor_name]['schedule']
            start_iso_time = get_iso_time(start, date, utc_offset=environment._utc_offset)
            assert isinstance(start, float) and isinstance(end, float) and isinstance(date, str) \
                and start < end and start >= self._START_HOUR and end <= self._END_HOUR \
                and compare_iso_time(start_iso_time, environment.current_time) and date in fixed_schedules
            assert gt_patient_condition['department'] == doctor_information[pred_doctor_name]['department']
            
            # Duration mismatched case
            if not float(Decimal(str(1)) / Decimal(str(doctor_information[pred_doctor_name]['capacity_per_hour']))) == float(Decimal(str(end)) - Decimal(str(start))):
                return False, STATUS_CODES['duration']
            
        except KeyError:
            return False, STATUS_CODES['format']    # Schedule allocation missing or doctor not found
        except AssertionError:
            return False, STATUS_CODES['schedule']    # Invalid schedule times or department

        ####################### Check the duplication of the schedules #######################
        prediction_schedule_segments = convert_time_to_segment(self._START_HOUR,
                                                               self._END_HOUR,
                                                               self._TIME_UNIT,
                                                               [start, end])
        fixed_schedule_segments = sum([convert_time_to_segment(self._START_HOUR, 
                                                               self._END_HOUR, 
                                                               self._TIME_UNIT, 
                                                               fs) for fs in fixed_schedules[date]], [])
        
        if len(set(prediction_schedule_segments) & set(fixed_schedule_segments)):
            return False, STATUS_CODES['conflict']['time']    # Overlaps with an existing schedule
        
        ####################### Check the patient's preferences  #######################
        if gt_patient_condition['preference'] == 'doctor':
            if gt_patient_condition.get('preferred_doctor') != pred_doctor_name:
                return False, STATUS_CODES['preference']['physician']
        
        if gt_patient_condition['preference'] == 'date':
            if compare_iso_time(gt_patient_condition.get('valid_from'), date):
                return False, STATUS_CODES['preference']['date']
        
        is_earliest = self.__check_is_earliest(
            prediction, 
            gt_patient_condition,
            doctor_information,
            environment,
        )

        if not is_earliest:
            return False, STATUS_CODES['preference']['asap']
        
        return True, STATUS_CODES['correct']


    def schedule_cancel(self, 
                        doctor_information: dict, 
                        environment: HospitalEnvironment, 
                        idx: Optional[int] = None, 
                        verbose: bool = False) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Cancel a doctor's scheduled appointment.

        Args:
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                       including availability and other relevant details.
            environment (HospitalEnvironment): Hospital environment.
            idx (int, optional): Specific patient schedule index.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.

        Returns:
            Tuple[Optional[dict], Optional[dict]]: Updated doctor information and a result dictionary after cancellation.
        """
        if idx is None:
            candidate_idx = [i for i, schedule in enumerate(environment.patient_schedules) if schedule['status'] == 'scheduled']
            idx = random.choice(candidate_idx) if len(candidate_idx) else -1

        if idx >= 0:
            # Ground-truth cancelled schedule
            cancelled_schedule = environment.patient_schedules[idx]
            patient = cancelled_schedule['patient']
            doctor, date, time = cancelled_schedule['attending_physician'], cancelled_schedule['date'], cancelled_schedule['schedule']
            
            # Initialize simulation environment for cancellation
            sim_environment = self._init_simulation(
                system_prompt_path=self.cancel_patient_system_prompt_path,
                environment=environment,
                additional_patient_conditions={
                    'patient_name': patient,
                    'doctor_name': doctor,
                    'date': date,
                    'start_time': hour_to_hhmmss(time[0])
                }
            )

            # Schedule cancellation simulation
            doctor_information, result_dict = run_with_retry(
                sim_environment.canceling_simulate,
                gt_idx=idx,
                doctor_information=doctor_information,
                patient_schedules=environment.patient_schedules,
                verbose=verbose,
                max_inferences=self.max_inferences,
                patient_kwargs=self.patient_reasoning_kwargs,
                staff_kwargs=self.staff_reasoning_kwargs,
                max_retries=self.max_retries,
            )

            # Successfully canceled
            if result_dict['status'][0] is not False:   # No GT and correct case
                # Update waiting list due to cancellation
                doctor_information, rs_result_dict = self.automatic_waiting_list_update(
                    sim_environment=sim_environment,
                    environment=environment,
                    doctor_information=doctor_information,
                )

                # Update result dictionary
                for key in result_dict.keys():
                    if len(rs_result_dict[key]):
                        result_dict[key].append(tuple(rs_result_dict[key]))
            
            return doctor_information, result_dict

        return None, None
                

    def rescheduling_request(self,
                             doctor_information: dict,
                             environment: HospitalEnvironment, 
                             idx: Optional[int] = None, 
                             verbose: bool = False) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Add a patient schedule to the waiting list in the given environment.

        Args:
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                        including availability and other relevant details.
            environment (HospitalEnvironment): Hospital environment.
            idx (int, optional): Specific patient schedule index.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.
        
        Returns:
            Tuple[Optional[dict], Optional[dict]]: Updated doctor information and a result dictionary after cancellation.
        """
        result_dict = init_result_dict()
        if idx is None:
            candidate_idx = [i for i, schedule in enumerate(environment.patient_schedules) if schedule['status'] == 'scheduled']
            idx = random.choice(candidate_idx) if len(candidate_idx) else -1
        
        if idx >= 0:
            requested_schedule = environment.patient_schedules[idx]
            if all(requested_schedule != s[1] for s in environment.waiting_list):
                # Ground-truth rescheduling requested schedule
                patient = requested_schedule['patient']
                doctor, date, time = requested_schedule['attending_physician'], requested_schedule['date'], requested_schedule['schedule']

                # Initialize simulation environment for rescheduling request
                sim_environment = self._init_simulation(
                    system_prompt_path=self.reschedule_patient_system_prompt_path,
                    environment=environment,
                    additional_patient_conditions={
                        'patient_name': patient,
                        'doctor_name': doctor,
                        'date': date,
                        'start_time': hour_to_hhmmss(time[0])
                    }
                )

                # Rescheduling request simulation
                try:
                    for event in sim_environment.rescheduling_simulate(
                        gt_idx=idx,
                        doctor_information=doctor_information,
                        patient_schedules=environment.patient_schedules,
                        verbose=verbose,
                        max_inferences=self.max_inferences,
                        patient_kwargs=self.patient_reasoning_kwargs,
                        staff_kwargs=self.staff_reasoning_kwargs,
                    ):
                        
                        if event['type'] == 'simulation':
                            prediction, original = event['prediction'], event['original']
                            status, status_code = self._sanity_check(
                                prediction,
                                original,
                                doctor_information,
                                environment
                            )

                            # When schedule sanity check is passed
                            if status:
                                sim_environment.update_from_kwargs(**{"_branch": True})
                            
                            # When schedule sanity check is failed
                            else:
                                result_dict = {
                                    'gt': [{'reschedule': idx}],
                                    'pred': [prediction],
                                    'status': [False],
                                    'status_code': [STATUS_CODES['reschedule']['schedule'].format(status_code=status_code)],
                                    'dialog': sim_environment.result_dict['dialog']
                                }
                                break
                        
                        # When succesfully rescheduled in the available time
                        else:
                            prediction, original = event['prediction'], event['original']
                            doctor_information[prediction['attending_physician']]['schedule'][prediction['date']].append(prediction['schedule'])
                            doctor_information[prediction['attending_physician']]['schedule'][prediction['date']].sort()
                            self.update_env(
                                status=status,
                                prediction=prediction,
                                environment=environment,
                            )
                            result_dict = {
                                'gt': [{'reschedule': idx}],
                                'pred': [prediction],
                                'status': [True],
                                'status_code': [STATUS_CODES['correct']],
                                'dialog': sim_environment.result_dict['dialog']
                            }
                            log(f'{colorstr("[RESCHEDULED]")}: {original} is rescheduled to {prediction}')

                    if not len(result_dict['gt']):
                        # Successfully adding to the waiting list
                        if len(sim_environment.result_dict['gt']):
                            result_dict = {
                                'gt': sim_environment.result_dict['gt'],
                                'pred': sim_environment.result_dict['pred'],
                                'status': sim_environment.result_dict['status'],
                                'status_code': sim_environment.result_dict['status_code'],
                                'dialog': sim_environment.result_dict['dialog']
                            }

                        # Not determined case during the simulation
                        else:
                            result_dict = {
                                'gt': [{'reschedule': idx}],
                                'pred': [None],
                                'status': [False],
                                'status_code': [STATUS_CODES['reschedule']['identify']],
                                'dialog': sim_environment.result_dict['dialog']
                            }
                
                # Requested schedule indentification error
                except ScheduleNotFoundError:
                    result_dict = {
                        'gt': sim_environment.result_dict['gt'],
                        'pred': sim_environment.result_dict['pred'],
                        'status': sim_environment.result_dict['status'],
                        'status_code': sim_environment.result_dict['status_code'],
                        'dialog': sim_environment.result_dict['dialog']
                    }

                # Tool calling error
                except TypeError:
                    result_dict = {
                        'gt': [{'reschedule': idx}],
                        'pred': [None],
                        'status': [False],
                        'status_code': [STATUS_CODES['reschedule']['type']],
                        'dialog': sim_environment.result_dict['dialog']
                    }
                
                # Otherwise
                except Exception as e:
                    status_code = f"Unexpected error: {e}"
                    log(status_code, level='warning')
                    result_dict = {
                        'gt': [{'reschedule': idx}],
                        'pred': [None],
                        'status': [False],
                        'status_code': [status_code],
                        'dialog': sim_environment.result_dict['dialog']
                    }

                return doctor_information, result_dict

            return None, None

        return None, None
            

    def automatic_waiting_list_update(self, 
                                      sim_environment: OPFVScheduleSimulation,
                                      environment: HospitalEnvironment,
                                      doctor_information: Optional[dict] = None) -> Tuple[dict, dict]:
        """
        Automatically update the waiting list by attempting to reschedule patients.

        Args:
            sim_environment (OPFVScheduleSimulation): The simulation environment used for scheduling.
            environment (HospitalEnvironment): Hospital environment.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details. Defaults to None.

        Returns:
            Tuple[dict, dict]: Updated doctor information and a result dictionary.
        """
        result_dict = init_result_dict()
        for turn, (idx, original) in enumerate(environment.waiting_list):
            if original['status'] == 'scheduled':
                filtered_doctor_information = environment.get_doctor_schedule(
                    doctor_information=doctor_information if not self.fhir_integration else None,
                    department=original['department'],
                    fhir_integration=self.fhir_integration,
                )
                _schedule_client = self.admin_staff_agent.build_agent(
                    rule=self.rules, 
                    doctor_info=filtered_doctor_information,
                    only_schedule_tool=True
                )
                prediction = sim_environment.scheduling(
                    client=_schedule_client,
                    known_condition=original,
                    doctor_information=doctor_information,
                    reschedule_flag=True,
                    **self.staff_reasoning_kwargs,
                )['result']

                status, status_code = self._sanity_check(
                    prediction, 
                    original,
                    doctor_information,
                    environment
                )

                if status:
                    pred_doctor_name = list(prediction['schedule'].keys())[0]
                    old_iso_time = get_iso_time(original['schedule'][0], original['date'])
                    new_iso_time = get_iso_time(prediction['schedule'][pred_doctor_name]['start'], prediction['schedule'][pred_doctor_name]['date'])
                    
                    if compare_iso_time(old_iso_time, new_iso_time):
                        doctor_information = self.rules.cancel_schedule(idx, doctor_information, original)
                        prediction = {
                            'patient': original['patient'],
                            'attending_physician': pred_doctor_name,
                            'department': original['department'],
                            'date': prediction['schedule'][pred_doctor_name]['date'],
                            'schedule': [
                                prediction['schedule'][pred_doctor_name]['start'], 
                                prediction['schedule'][pred_doctor_name]['end']
                            ],
                            'patient_intention': original['patient_intention'],
                            'preference': original.get('preference'),
                            'preferred_doctor': original.get('preferred_doctor'),
                            'valid_from': original.get('valid_from'),
                            'last_updated_time': environment.current_time
                        }
                        doctor_information[prediction['attending_physician']]['schedule'][prediction['date']].append(prediction['schedule'])
                        doctor_information[prediction['attending_physician']]['schedule'][prediction['date']].sort()
                        self.update_env(
                            status=status,
                            prediction=prediction,
                            environment=environment,
                        )
                        result_dict['gt'].append('automatic rescheduling')
                        result_dict['pred'].append(prediction)
                        result_dict['status'].append(True)
                        result_dict['status_code'].append(STATUS_CODES['correct'])
                        result_dict['dialog'].append('automatic waiting list update from the system')
                        log(f'{colorstr("[RESCHEDULED]")}: {original} is rescheduled to {prediction}')
                
                else:
                    result_dict['gt'].append('automatic rescheduling')
                    result_dict['pred'].append(prediction)
                    result_dict['status'].append(status)
                    result_dict['status_code'].append(STATUS_CODES['reschedule']['schedule'].format(status_code=status_code))
                    result_dict['dialog'].append('automatic waiting list update from the system')
        
        return doctor_information, result_dict
                   

    def update_env(self, 
                   status: bool, 
                   prediction: Union[dict, str], 
                   environment: HospitalEnvironment, 
                   patient_information: Optional[dict] = None):
        """
        Update the simulation environment with scheduling results and optionally synchronize FHIR resources.

        Args:
            status (bool): Whether the scheduling task was successful. If True, FHIR resources may be updated.
            prediction (Union[dict, str]): The predicted scheduling result (e.g., patient schedule information).
            environment (HospitalEnvironment): The environment instance to be updated (must implement `update_env`).
            patient_information (Optional[dict], optional): Patient-related predicted (or GT) information to generate FHIR Patient resources. Defaults to None.

        """
        # POST/PUT to FHIR
        fhir_patient, fhir_appointment = None, None
        if status and self.fhir_integration:
            if patient_information is not None:
                fhir_patient = DataConverter.data_to_patient(
                    {
                        'metadata': deepcopy(self._metadata),
                        'department': deepcopy(self._department_data),
                        'patient': {
                            prediction['patient']: {
                                'department': prediction['department'], 
                                'gender': patient_information['gender'],
                                'telecom': [{'system': 'phone', 'value': patient_information['phone_number'], 'use': 'mobile'}],
                                'birthDate': personal_id_to_birth_date(patient_information['personal_id']),
                                'identifier': [{'value': patient_information['personal_id'], 'use': 'official'}],
                                'address': [{'type': 'postal', 'text': patient_information['address'], 'use': 'home'}],
                            }
                        }
                    }
                )[0]
            fhir_appointment = DataConverter.get_fhir_appointment(data={'metadata': deepcopy(self._metadata),
                                                                        'department': deepcopy(self._department_data),
                                                                        'information': deepcopy(prediction)})
            
        environment.update_env(
            status=status, 
            patient_schedule=prediction,
            fhir_resources={'Patient': fhir_patient, 'Appointment': fhir_appointment}
        )
            

    def __call__(self, data_pair: Tuple[dict, dict], agent_test_data: dict, agent_results: dict, environment, verbose: bool = False) -> dict:
        """
        This method uses agent test data to prompt an LLM for scheduling decisions, post-processes
        the output, runs sanity checks on predicted schedules, and collects the results for evaluation.

        Args:
            data_pair (Tuple[dict, dict]): A pair of ground truth and patient data for agent simulation.
            agent_test_data (dict): Dictionary containing test data and metadata for a single hospital.
                Expected keys include:
                    - 'metadata': A dict containing start_hour, end_hour, and interval_hour under 'time'.
                    - 'agent_data': A list of (ground_truth, test_data) pairs.
                    - 'doctor': A dictionary of doctor profiles with department and schedule info.
            agent_results (dict): Optional dictionary containing prior department predictions.
                Used to extract department-level guidance per patient. Can be empty.
            environment (HospitalEnvironment): Hospital environment instance to manage patient schedules.
            verbose (bool, option): Whether logging the each result or not.

        Returns:
            dict: A dictionary with three keys:
                - 'gt': List of ground truth results, each including patient info, attending physician, department, and schedule.
                - 'pred': List of predicted results (either valid dict or fallback string).
                - 'status': List of booleans indicating whether each prediction passed sanity checks.
                - 'status_code': List of status codes explaining each status.
        """
        gt, test_data = data_pair
        self._metadata = agent_test_data.get('metadata')
        self._department_data = agent_test_data.get('department')
        self._START_HOUR = self._metadata.get('time').get('start_hour')
        self._END_HOUR = self._metadata.get('time').get('end_hour')
        self._TIME_UNIT = self._metadata.get('time').get('interval_hour')
        self._DAY = self._metadata.get('days')
        doctor_information = environment.get_general_doctor_info_from_fhir() if self.fhir_integration else agent_test_data.get('doctor')
        patient_info, department, sanity = self.get_intake_information(gt, agent_results, doctor_information)
        self.rules = SchedulingRule(self._metadata, self._department_data, environment, self.fhir_integration)
        results = init_result_dict()

        # Make scheduling GT list
        gt_data = [
            {
                'patient': patient_info['name'] if sanity else gt.get('patient'),
                'department': department if sanity else gt.get('department'),
                'preference': preference,
                'preferred_doctor': gt.get('attending_physician') if preference == 'doctor' else "Doesn't matter",
                'valid_from': gt.get('valid_from') if preference == 'date' else None,
            } for preference in gt.get('preference')
        ]
        staff_known_data = [
            {
                'patient': patient_info['name'],
                'department': department,
                'patient_intention': None,
            } for _ in range(len(gt_data))
        ]

        # If the precedent department data is wrong, continue
        if not sanity:
            results['gt'].append(gt_data)
            results['pred'].append({})
            results['status'].append(False)
            results['status_code'].append(STATUS_CODES['preceding'])
            return results
        
        # Initialize the simulation environment using the first preference data
        preference = gt_data[0].get('preference')
        preference_desc = PREFERENCE_PHRASE_PATIENT[preference] if preference != 'date' \
                    else PREFERENCE_PHRASE_PATIENT[preference].format(date=gt_data[0].get('valid_from'))
        sim_environment = self._init_simulation(
            system_prompt_path=self.schedule_patient_system_prompt_path,
            environment=environment,
            additional_patient_conditions={
                'preference': preference,
                'preference_desc': preference_desc,
                'preferred_doctor': gt_data[0]['preferred_doctor'],
            }
        )
    
        # Simulate the main scheduling task
        # TODO: using retry function
        for i, prediction in enumerate(sim_environment.scheduling_simulate(
            gt_data=gt_data,
            staff_known_data=staff_known_data,
            doctor_information=doctor_information,
            verbose=verbose,
            patient_kwargs=self.patient_reasoning_kwargs,
            staff_kwargs=self.staff_reasoning_kwargs
        )):
            status, status_code = self._sanity_check(
                prediction, 
                gt_data[i],
                doctor_information,
                environment
            )

            if status:
                pred_doctor_name = list(prediction['schedule'].keys())[0]
                prediction = {
                    'patient': staff_known_data[i]['patient'],
                    'attending_physician': pred_doctor_name,
                    'department': staff_known_data[i]['department'],
                    'date': prediction['schedule'][pred_doctor_name]['date'],
                    'schedule': [
                        prediction['schedule'][pred_doctor_name]['start'], 
                        prediction['schedule'][pred_doctor_name]['end']
                    ],
                    'patient_intention': staff_known_data[i]['patient_intention'],
                    'preference': gt_data[i].get('preference'),
                    'preferred_doctor': gt_data[i].get('preferred_doctor'),
                    'valid_from': gt_data[i].get('valid_from'),
                    'last_updated_time': environment.current_time
                }
            else:
                break
        
        if verbose:
            log(f'Pred  : {prediction}')
            log(f'Status: {status_code}')
            log(f'Final Status: {status_code}\n\n\n')

        # Update the simulation environment and the doctor information in the agent test data
        if status:
            doctor_information[prediction['attending_physician']]['schedule'][prediction['date']].append(prediction['schedule'])
            doctor_information[prediction['attending_physician']]['schedule'][prediction['date']].sort()
        
        self.update_env(
            status=status,
            prediction=prediction,
            environment=environment,
            patient_information=patient_info,
        )
        agent_test_data['doctor'] = doctor_information

        # Append results
        results['gt'].append(gt_data[i])
        results['pred'].append(prediction)
        results['status'].append(status)
        results['status_code'].append(status_code)
        results['dialog'].append(sim_environment.result_dict['dialog'])
        
        # Other events
        ## Simulate the schedule cancellation requests
        if random.random() < self.schedule_cancellation_prob:
            updated_doctor_information, result_dict = self.schedule_cancel(
                doctor_information=doctor_information,
                environment=environment,
                verbose=verbose,
            )
            if updated_doctor_information is not None:
                agent_test_data['doctor'] = updated_doctor_information
                results['gt'].extend(result_dict['gt'])
                results['pred'].extend(result_dict['pred'])
                results['status'].extend(result_dict['status'])
                results['status_code'].extend(result_dict['status_code'])
                results['dialog'].extend(result_dict['dialog'])

                if verbose:
                    log(f'Pred  : {result_dict["pred"]}')
                    log(f'Status: {result_dict["status_code"]}')
                    log(f'Final Status: {result_dict["status_code"]}\n\n\n')
        
        ## Simulate the resecheduling requests
        if random.random() < self.request_early_schedule_prob:
            updated_doctor_information, result_dict = self.rescheduling_request(
                doctor_information=doctor_information,
                environment=environment, 
                verbose=verbose
            )
            if updated_doctor_information is not None:
                agent_test_data['doctor'] = updated_doctor_information
                results['gt'].extend(result_dict['gt'])
                results['pred'].extend(result_dict['pred'])
                results['status'].extend(result_dict['status'])
                results['status_code'].extend(result_dict['status_code'])
                results['dialog'].extend(result_dict['dialog'])

                if verbose:
                    log(f'Pred  : {result_dict["pred"]}')
                    log(f'Status: {result_dict["status_code"]}')
                    log(f'Final Status: {result_dict["status_code"]}\n\n\n')


        # TODO: token stats

        return results
