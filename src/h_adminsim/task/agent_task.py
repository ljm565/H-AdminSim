import os
import json
import time
import random
from copy import deepcopy
from importlib import resources
from openai import InternalServerError
from decimal import Decimal, getcontext
from typing import Tuple, Union, Optional
from google.genai.errors import ServerError
from dotenv import load_dotenv, find_dotenv

from patientsim import PatientAgent
from patientsim import AdminStaffAgent as IntakeAdminStaffAgent
from patientsim.environment import OPSimulation as OPFVIntakeSimulation

from h_adminsim import SupervisorAgent
from h_adminsim import AdminStaffAgent as SchedulingAdminStaffAgent
from h_adminsim.environment.hospital import HospitalEnvironment
from h_adminsim.registry import STATUS_CODES, SCHEDULING_ERROR_CAUSE
from h_adminsim.tools import DataConverter, SchedulingRule, scheduling_tool_calling
from h_adminsim.utils import colorstr, log
from h_adminsim.utils.fhir_utils import *
from h_adminsim.utils.filesys_utils import txt_load, json_load
from h_adminsim.utils.common_utils import (
    group_consecutive_segments,
    personal_id_to_birth_date,
    convert_segment_to_time,
    convert_time_to_segment,
    exponential_backoff,
    compare_iso_time,
    get_iso_time,
)



class FirstVisitOutpatientTask:
    def __init__(self):
        self.token_stats = {
            'patient_token': {'input':[], 'output': [], 'reasoning': []}, 
            'admin_staff_token': {'input': [], 'output': [], 'reasoning': []}, 
            'supervisor_token': {'input':[], 'output': [], 'reasoning': []}
        }


    def get_result_dict(self) -> dict:
        """
        Initialize result dictionary.

        Returns:
            dict: Initialized result dictionary.
        """
        return {'gt': [], 'pred': [], 'status': [], 'status_code': [], 'trial': [], 'dialog': []}
    

    def run_with_retry(self, func, *args, max_retries=8, **kwargs):
        retry_count = 0

        while 1:
            try:
                return func(*args, **kwargs)

            except (ServerError, InternalServerError) as e:
                if retry_count >= max_retries:
                    log(f"\nMax retries reached. Last error: {e}", level='error')
                    raise e

                wait_time = exponential_backoff(retry_count)
                log(
                    f"[{retry_count + 1}/{max_retries}] {type(e).__name__}: {e}. "
                    f"Retrying in {wait_time:.1f} seconds...",
                    level='warning',
                )
                time.sleep(wait_time)
                retry_count += 1

    
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
    

    def _get_fhir_appointment(self,
                              gt_resource_path: Optional[str] = None,
                              data: Optional[dict] = None) -> dict:
        """
        Load a FHIR Appointment resource from a file path if available, or generate it dynamically from the provided data.

        Args:
            gt_resource_path (Optional[str], optional):  
                Path to the ground-truth FHIR Appointment resource file.  
                If the file exists, it will be loaded and returned.  
                If not, a resource will be generated from the `data` argument.
            data (Optional[dict], optional):  
                Dictionary containing the metadata and patient information  
                needed to generate the Appointment resource.  
                Expected to include 'metadata' and 'information' keys.

        Returns:
            dict: A FHIR Appointment resource in dictionary form.
        """
        try:
            return json_load(gt_resource_path)
        except:
            metadata, info, department = data.get('metadata'), data.get('information'), data.get('department')
            schedule = info.get('schedule')
            if 'time' in schedule:
                schedule = schedule.get('time')
            
            gt_resource = DataConverter.data_to_appointment(
                {
                    'metadata': metadata,
                    'department': department,
                    'patient': {
                        info.get('patient'): {
                            'department': info.get('department'),
                            'attending_physician': info.get('attending_physician'),
                            'date': info.get('date'),
                            'schedule': schedule
                        }
                    }
                }
            )[0]
            return gt_resource
        
    
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
        results = self.get_result_dict()
        
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
        environment = OPFVIntakeSimulation(patient_agent, admin_staff_agent, max_inferences=self.max_inferences)
        output = self.run_with_retry(
                environment.simulate,
                verbose=False,
                patient_kwargs=self.patient_reasoning_kwargs, 
                staff_kwargs=self.staff_reasoning_kwargs,
                max_retries=self.max_retries,
            )
        dialogs, patient_token, admin_staff_token = output['dialog_history'], output.get('patient_token_usage'), output.get('admin_staff_token_usage')
        prediction_department = OutpatientFirstIntake.postprocessing_department(dialogs[-1]['content'])

        # LLM call: Agent which should extract demographic information of the patient and evaluation the department decision result
        dialogs = '\n'.join([f"{turn['role']}: {' '.join(turn['content'].split())}" for turn in dialogs])
        
        if self.use_supervisor:
            user_prompt = self.supervisor_client.user_prompt_template.format(
                CONVERSATION=dialogs,
                DEPARTMENTS=''.join([f'{i+1}. {department}\n' for i, department in enumerate(departments)])
            )
            prediction_supervision = self.run_with_retry(
                self.supervisor_client,
                user_prompt,
                using_multi_turn=False,
                verbose=False,
                max_retries=self.max_retries,
            )
        else:
            prediction_supervision = self.run_with_retry(
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
                 scheduling_strategy: str,
                 schedule_cancellation_prob: float = 0.05,
                 request_early_schedule_prob: float = 0.1,
                 preference_rejection_prob: float = 0.3,
                 preferene_rejection_prob_decay: float = 0.5,
                 fhir_integration: bool = False,
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
        
        # Scheduling strategy and feedback mechanism
        self.scheduling_strategy = scheduling_strategy
        assert self.scheduling_strategy in ['llm', 'tool_calling'], \
            log('Scheduling strategy must be either "llm" or "tool_calling".', 'error')
        
        # Initialize scheduling methods and a staff agent
        self.admin_staff_agent = SchedulingAdminStaffAgent(
            target_task='first_outpatient_scheduling',
            model=self.admin_staff_model,
            use_vllm=self.admin_staff_use_vllm,
            vllm_endpoint=self.admin_staff_vllm_endpoint
        )
        log(f'Scheduling strategy: {colorstr(self.scheduling_strategy)}')

        # Scheduling parameters
        self.schedule_cancellation_prob = schedule_cancellation_prob
        self.request_early_schedule_prob = request_early_schedule_prob
        self.preference_rejection_prob = preference_rejection_prob
        self.preferene_rejection_prob_decay = preferene_rejection_prob_decay

        # Others
        self.fhir_integration = fhir_integration
        self.max_retries = max_retries
        self.patient_system_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath('schedule_patient_system.txt'))
        self.patient_rejected_system_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath('schedule_patient_rejected_system.txt'))
        self.preference_phrase_patient = {
            'asap': 'You want the earliest available doctor in the department for the outpatient visit.',
            'doctor': 'You have a preferred doctor for the outpatient visit.',
            'date': 'You want the earliest available doctor in the department for the outpatient visit, starting from **{date}**.'
        }
        self.preference_phrase_staff = {
            'asap': 'The patient wants the earliest available doctor in the department for the outpatient visit.',
            'doctor': 'The patient has a preferred doctor for the outpatient visit.',
            'date': 'The patient wants the earliest available doctor in the department for the outpatient visit, starting from **{date}**.'
        }
        self.schedule_suggestion_desc = "How about this schedule: {schedule}"

    
    @staticmethod
    def postprocessing(text: Union[str, dict]) -> Union[str, dict]:
        """
        Attempts to parse the given text as JSON. If parsing succeeds, returns a dictionary;
        otherwise, returns the original string.

        Args:
            text (Union[str, dict]): The text output to post-process, potentially a JSON-formatted string. 

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
            
            assert len(text_dict) == 1 and all(k in text_dict for k in ['schedule'])   # Basic sanity check
            key = list(text_dict['schedule'].keys())[0]
            text_dict['schedule'][key]['start'] = float(text_dict['schedule'][key]['start'])
            text_dict['schedule'][key]['end'] = float(text_dict['schedule'][key]['end'])
            text_dict['schedule'][key]['date'] = str(text_dict['schedule'][key]['date'])
            return text_dict
        
        except:
            return str(text)
    

    def __init_patient_agent(self, 
                             patient_condition: dict,
                             patient_system_prompt_path: str,
                             **kwargs) -> PatientAgent:
        """
        Initialize patient agent.

        Args:
            patient_condition (dict): The conditions including name, preference, etc.
            patient_system_prompt_path (str): The system prompt path for the patient agent.

        Returns:
            PatientAgent: Initialized patient agent.
        """
        preference = patient_condition.get('preference')
        preference_desc = self.preference_phrase_patient[preference] if preference != 'date' \
                    else self.preference_phrase_patient[preference].format(date=patient_condition.get('valid_from'))
        
        patient_agent = PatientAgent(
            self.patient_model,
            'outpatient',
            lang_proficiency_level='B',
            system_prompt_path=patient_system_prompt_path,
            log_verbose=False,
            additional_patient_conditions={
                'preference': preference,
                'preference_desc': preference_desc,
                'preferred_doctor': patient_condition['preferred_doctor'],
                **kwargs
            },
            temperature=0 if not 'gpt-5' in self.patient_model.lower() else 1
        )
        return patient_agent


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
        
        # ###################### Check the doctors' workload balance  #####################
        # if not patient_condition['preference'] == 'doctor':
        #     schedule_candidates = self.__filter_doctor_schedule(
        #         doctor_information,
        #         patient_condition.get('department'),
        #         environment
        #     )['doctor']
        #     selected_doctor_wl = float(schedule_candidates[doctor_name]['workload'][:-1])
        #     if not all(float(v['workload'][:-1]) >= selected_doctor_wl for k, v in schedule_candidates.items() if k != doctor_name):
        #         return False, STATUS_CODES['workload']
                
        return True, STATUS_CODES['correct']


    def schedule_cancel(self, 
                        doctor_information: dict, 
                        environment: HospitalEnvironment, 
                        idx: Optional[int] = None, 
                        verbose: bool = False) -> dict:
        """
        Cancel a doctor's scheduled appointment.

        Args:
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                       including availability and other relevant details.
            environment (HospitalEnvironment): Hospital environment.
            idx (int, optional): Specific patient schedule index.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.

        Returns:
            dict: The updated doctor information with the cancelled schedule removed.
        """
        if not idx:
            candidate_idx = [i for i, schedule in enumerate(environment.patient_schedules) if schedule['status'] == 'scheduled']
            idx = random.choice(candidate_idx) if len(candidate_idx) else -1

        if idx >= 0:
            cancelled_schedule = environment.patient_schedules[idx]
            doctor, date, time = cancelled_schedule['attending_physician'], cancelled_schedule['date'], cancelled_schedule['schedule']
            schedule_list = doctor_information[doctor]['schedule'][date]
            schedule_list.remove(time)

            if self.fhir_integration:
                fhir_appointment = self._get_fhir_appointment(data={'metadata': deepcopy(self._metadata),
                                                                    'department': deepcopy(self._department_data),
                                                                    'information': deepcopy(cancelled_schedule)})
                environment.delete_fhir({'Appointment': fhir_appointment})

            environment.schedule_cancel_event(idx, verbose)

        return doctor_information
    

    def move_up_schedule(self,
                         doctor_information: dict,
                         environment: HospitalEnvironment,
                         verbose: bool = False) -> Tuple[list[bool], list[str], list[Union[str, dict]], dict]:
        """
        Move up the patient's schedule in the waiting list.

        Args:
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                       including availability and other relevant details.
            environment (HospitalEnvironment): Hospital environment.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.

        Returns:
            Tuple[list[bool], list[str], list[Union[str, dict]], dict, list[list[str]]]: 
                - Multiple results of boolean indicating whether the prediction passed all sanity checks.
                - Multiple results of string explaining its status.
                - Multiple results of the original prediction (wrong case) or processed prediction (correct case) (either unchanged or used for debugging/logging if invalid).
                - Updated doctor information after processing the prediction.
        """
        statuses, status_codes, predictions = list(), list(), list()
        for turn, (idx, schedule) in enumerate(environment.waiting_list):
            if schedule['status'] == 'scheduled':
                is_earliest = self.__check_is_earliest(
                    prediction={
                        'schedule': {
                            schedule['attending_physician']: {
                                'date': schedule['date'],
                                'start': schedule['schedule'][0],
                                'end': schedule['schedule'][1],
                            }
                        }
                    },
                    gt_patient_condition={
                        'department': schedule['department'],
                        'preference': schedule['preference'],
                        'valid_from': schedule['valid_from'],
                    },
                    doctor_information=doctor_information,
                    environment=environment,
                )
                if not is_earliest:
                    status, status_code, prediction = self.scheduling(
                        known_condition={
                            'patient': schedule['patient'],
                            'department': schedule['department'],
                            'patient_intention': schedule['patient_intention']
                        },
                        gt_patient_condition={
                            'patient': schedule['patient'],
                            'department': schedule['department'],
                            'preference': schedule['preference'],
                            'preferred_doctor': schedule['preferred_doctor'],
                            'valid_from': schedule['valid_from'],
                        },
                        doctor_information=doctor_information,
                        environment=environment,
                        reschedule_flag=True,
                        verbose=verbose
                    )

                    if status:
                        statuses.append(status)
                        status_codes.append(status_code)
                        predictions.append(prediction)
                        doctor_information = self.schedule_cancel(doctor_information, environment, idx)
                        self.update_env(status, prediction, environment)    # Only update appointment information because patient information already exists
                    else:
                        statuses.append(status)
                        status_codes.append(STATUS_CODES['moving up schedule'])
                        predictions.append(prediction)
        
        return statuses, status_codes, predictions, doctor_information
    

    def add_waiting_list(self, 
                         environment: HospitalEnvironment, 
                         idx: Optional[int] = None, 
                         verbose: bool = False):
        """
        Add a patient schedule to the waiting list in the given environment.

        Args:
            environment (HospitalEnvironment): Hospital environment.
            idx (int, optional): Specific patient schedule index.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.
        """
        if not idx:
            candidate_idx = [i for i, schedule in enumerate(environment.patient_schedules) if schedule['status'] == 'scheduled']
            idx = random.choice(candidate_idx) if len(candidate_idx) else -1
        
        if idx >= 0:
            environment.add_waiting_list(idx, verbose)


    def scheduling(self,
                   known_condition: dict,
                   gt_patient_condition: dict, 
                   doctor_information: dict, 
                   environment: HospitalEnvironment, 
                   reschedule_flag: bool = False, 
                   verbose: bool = False) -> Tuple[bool, str, Union[str, dict]]:
        """
        Make an appointment between the doctor and the patient.

        Args:
            known_condition (dict): Patient conditions known to the staff.
            gt_patient_condition (dict): Ground truth patient conditions used only for sanity checks.
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                       including availability and other relevant details.
            environment (HospitalEnvironment): Hospital environment.
            reschedule_flag (bool, optional): Whether this process is rescheduling or not. Defaults to False.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.

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
        if self.scheduling_strategy == 'llm':
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
            prediction = self.run_with_retry(
                self.admin_staff_agent,
                user_prompt,
                using_multi_turn=False,
                verbose=False,
                max_retries=self.max_retries,
            )
            prediction = OutpatientFirstScheduling.postprocessing(prediction)    
            status, status_code = self._sanity_check(
                prediction, 
                gt_patient_condition,
                doctor_information,
                environment
            )

            if status:
                pred_doctor_name = list(prediction['schedule'].keys())[0]
                prediction = {
                    'patient': known_condition['patient'],
                    'attending_physician': pred_doctor_name,
                    'department': known_condition['department'],
                    'date': prediction['schedule'][pred_doctor_name]['date'],
                    'schedule': [
                        prediction['schedule'][pred_doctor_name]['start'], 
                        prediction['schedule'][pred_doctor_name]['end']
                    ],
                    'patient_intention': known_condition['patient_intention'],
                    'preference': gt_patient_condition.get('preference'),
                    'preferred_doctor': gt_patient_condition.get('preferred_doctor'),
                    'valid_from': gt_patient_condition.get('valid_from'),
                    'last_updated_time': environment.current_time
                }

            if verbose: 
                log(f'Pred  : {prediction}')
                log(f'Status: {status_code}')
                
            # Append token data and reset agents
            self.save_token_data(
                admin_staff_token=self.admin_staff_agent.client.token_usages, 
            )
            self.admin_staff_agent.reset_history(verbose=False)
        
        ############################# Tool calling-based Scheduling ############################
        elif self.scheduling_strategy == 'tool_calling':
            filtered_doctor_information = environment.get_doctor_schedule(
                doctor_information=doctor_information if not self.fhir_integration else None,
                department=department,
                fhir_integration=self.fhir_integration,
            )
            self.client = self.admin_staff_agent.build_agent(self.rules, filtered_doctor_information)
            try:
                prediction = scheduling_tool_calling(self.client, self.rules, known_condition, doctor_information, environment.current_time)
                status, status_code = self._sanity_check(
                    prediction, 
                    gt_patient_condition,
                    doctor_information,
                    environment
                )
            except:
                log('Fail to load an appropriate tool', level='warning')
                prediction = 'Fail to load an appropriate tool'
                status, status_code  = False, STATUS_CODES['tool']
                    
            if status:
                pred_doctor_name = list(prediction['schedule'].keys())[0]
                prediction = {
                    'patient': known_condition['patient'],
                    'attending_physician': pred_doctor_name,
                    'department': known_condition['department'],
                    'date': prediction['schedule'][pred_doctor_name]['date'],
                    'schedule': [
                        prediction['schedule'][pred_doctor_name]['start'], 
                        prediction['schedule'][pred_doctor_name]['end']
                    ],
                    'patient_intention': known_condition['patient_intention'],
                    'preference': gt_patient_condition.get('preference'),
                    'preferred_doctor': gt_patient_condition.get('preferred_doctor'),
                    'valid_from': gt_patient_condition.get('valid_from'),
                    'last_updated_time': environment.current_time
                }

            if verbose: 
                log(f'Pred  : {prediction}')
                log(f'Status: {status_code}') 

        else:
            raise NotImplementedError(
                colorstr('red', 'Unsupported strategy. Supported strategies are ["llm", "tool_calling"].')
            )

        return status, status_code, prediction
    

    def get_intention(self, 
                      patient_condition: dict, 
                      rejected_preference: Optional[str] = None,
                      rejected_schedule: Optional[dict] = None) -> str:
        """
        Get patient's scheduling preferences fron conversation.

        Args:
            patient_condition (dict): Patient characteristics, including the ground-truth scheduling preference.
            rejected_preference (Optional[str], optional): Scheduling preference rejected in the previous turn.
            rejected_schedule (Optional[dict], optional): Scheduling proposal previously suggested by the staff and rejected by the patient.

        Returns:
            str: The patient utterance having his intention.
        """
        if rejected_preference:
            rejected_preference_desc = self.preference_phrase_staff[rejected_preference] if rejected_preference != 'date' \
                    else self.preference_phrase_staff[rejected_preference].format(date='a specific date')
            self.patient_agent = self.__init_patient_agent(
                patient_condition=patient_condition, 
                patient_system_prompt_path=self.patient_rejected_system_prompt_path,
                rejected_preference=rejected_preference_desc
            )
            user_prompt = self.schedule_suggestion_desc.format(schedule=rejected_schedule)
        else:
            self.patient_agent = self.__init_patient_agent(
                patient_condition=patient_condition, 
                patient_system_prompt_path=self.patient_system_prompt_path
            )
            user_prompt = 'How would you like to schedule the appointment?'
        
        role = f"{colorstr('blue', 'Staff')}"
        log(f"{role:<25}: {user_prompt}")

        # Patient response
        response = self.run_with_retry(
            self.patient_agent,
            user_prompt,
            using_multi_turn=False,
            verbose=False,
            max_retries=self.max_retries,
        )
        role = f"{colorstr('green', 'Patient')} ({patient_condition['preference']})"
        log(f"{role:<25}: {response}")

        # Save token data and reset the staff agent
        self.save_token_data(
            patient_token=self.patient_agent.client.token_usages,
        )

        # Reset the staff agent history for the scheduling task
        self.admin_staff_agent.reset_history(verbose=False)        

        return response
    

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
            fhir_appointment = self._get_fhir_appointment(data={'metadata': deepcopy(self._metadata),
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
        self.rules = SchedulingRule(self._metadata, environment)
        results = self.get_result_dict()

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
        
        # Iterate over multiple preferences if exists
        preference_reject_prob = 0.0 if len(gt_data) <= 1 else self.preference_rejection_prob
        for i, (gt_patient_condition, staff_known_condition) in enumerate(zip(gt_data, staff_known_data)):
            # Get patient's intention
            patient_intention_utterance = self.get_intention(
                patient_condition=gt_patient_condition,
                rejected_preference=None if i == 0 else gt_data[i-1]['preference'],
                rejected_schedule=None if i == 0 else prediction,
            )
            staff_known_condition.update({'patient_intention': patient_intention_utterance})

            # Scheduling
            status, status_code, prediction = self.scheduling(
                staff_known_condition,
                gt_patient_condition,
                doctor_information,
                environment,
                verbose=verbose,
            )
            if not status:
                break

            # Preference rejection logic
            ## Rejection case
            if random.random() < preference_reject_prob and i != len(gt_data) - 1:
                preference_reject_prob *= self.preferene_rejection_prob_decay
            ## Non-rejection case
            else:
                break

        if verbose:
            if status:
                role = f"{colorstr('blue', 'Staff')}"
                log(f"{role:<25}: {self.schedule_suggestion_desc.format(schedule=prediction)}")
                role = f"{colorstr('green', 'Patient')} ({gt_patient_condition['preference']})"
                log(f"{role:<25}: Thank you.")
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
        results['gt'].append(gt_patient_condition)
        results['pred'].append(prediction)
        results['status'].append(status)
        results['status_code'].append(status_code)
        
        # Other events
        ## Schedule cancellation
        if random.random() < self.schedule_cancellation_prob:
            agent_test_data['doctor'] = self.schedule_cancel(
                doctor_information=doctor_information,
                environment=environment,
                verbose=verbose,
            )
        
        ## Try to move up the existing patient schedule
        if random.random() < self.request_early_schedule_prob:
            self.add_waiting_list(environment, verbose=verbose)

        ## Regular check of the waiting list 
        statuses, status_codes, predictions, doctor_information = self.move_up_schedule(
            doctor_information=doctor_information,
            environment=environment,
            verbose=verbose,
        )
        agent_test_data['doctor'] = doctor_information
        results['pred'].extend(predictions)
        results['status'].extend(statuses)
        results['status_code'].extend(status_codes)
        results['gt'].extend(deepcopy(predictions))     # To syncronize the number of each result

        return results
