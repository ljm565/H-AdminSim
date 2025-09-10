import json
from copy import deepcopy
from typing import Tuple, Union, Optional
from patientsim import AdminStaffAgent, PatientAgent
from patientsim.environment import OPSimulation

from tools import (
    GeminiClient,
    GeminiLangChainClient,
    GPTClient,
    GPTLangChainClient,
    VLLMClient,
    DataConverter,
)
from registry import STATUS_CODES, SCHEDULING_ERROR_CAUSE
from utils import log
from utils.fhir_utils import *
from utils.filesys_utils import txt_load, json_load
from utils.common_utils import (
    convert_segment_to_time,
    convert_time_to_segment,
    compare_iso_time,
    get_iso_time,
)



class Task:
    def get_result_dict(self) -> dict:
        """
        Initialize result dictionary.

        Returns:
            dict: Initialized result dictionary.
        """
        return {'gt': [], 'pred': [], 'status': [], 'status_code': []}
    

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
        
    
    # TODO: deprecated function
    def sanity_check_recursively(self, prediction: dict, expected_prediction: dict) -> bool:
        """
        Recursively compares two nested data structures (dictionaries and lists) 
        to check if the `prediction` matches the `expected_prediction` in structure and values.
        The comparison is order-insensitive for lists, meaning list elements can appear in any order as long as all expected items are matched.

        Args:
            prediction (dict): The data structure to validate, typically the output or result.
            expected_prediction (dict): The reference data structure to compare against.

        Returns:
            bool: True if `prediction` matches `expected_prediction` recursively in structure and content;  
                  False otherwise.
        """
        if isinstance(expected_prediction, dict):
            if not isinstance(prediction, dict):
                return False
            for key, val in expected_prediction.items():
                if key not in prediction:
                    return False
                if not self.sanity_check_recursively(prediction[key], val):
                    return False
            return True
        
        elif isinstance(expected_prediction, list):
            if not isinstance(prediction, list) or len(expected_prediction) != len(prediction):
                return False
            
            unmatched = prediction.copy()
            for expected_item in expected_prediction:
                matched = False
                for i, pred_item in enumerate(unmatched):
                    if self.sanity_check_recursively(pred_item, expected_item):
                        unmatched.pop(i)
                        matched = True
                        break
                if not matched:
                    return False
            return True

        else:
            return expected_prediction == prediction



class OutpatientIntake(Task):
    def __init__(self, config):
        # Initialize variables
        self.name = 'intake'
        self.supervisor_model = config.supervisor_model
        self.task_model = config.task_model
        self._sup_system_prompt_path = config.outpatient_intake.supervisor_system_prompt
        self._sup_user_prompt_path = config.outpatient_intake.supervisor_user_prompt
        self.ensure_output_format = config.ensure_output_format
        
        # Initialize prompts
        self.system_prompt = txt_load(self._sup_system_prompt_path)
        self.user_prompt_template = txt_load(self._sup_user_prompt_path)

        # Initialize models
        if 'gemini' in self.supervisor_model.lower():
            self.supervisor_client = GeminiLangChainClient(self.supervisor_model) if self.ensure_output_format else GeminiClient(self.supervisor_model)
        elif 'gpt' in self.supervisor_model.lower():
            self.supervisor_client = GPTLangChainClient(self.supervisor_model) if self.ensure_output_format else GPTClient(self.supervisor_model)
        else:
            self.supervisor_client = VLLMClient(self.supervisor_model, config.vllm_url)

    
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
                if not match:
                    return text
                json_str = match.group(1)
                text_dict = json.loads(json_str)
                
            else:
                text_dict = text
            
            assert len(text_dict) == 4 and all(k in text_dict for k in ['name', 'birth_date', 'gender', 'department'])   # Basic sanity check
            return text_dict
        except:
            return str(text)
        
    
    def _department_decision(self, prediction_department: str, prediction_supervison: Union[str, dict]) -> str:
        """
        Determine the final department decision by considering both 
        the interaction agent result and the supervisor agent result.

        Args:
            prediction_department (str): The department predicted by the interaction agent.
            prediction_supervison (Union[str, dict]): The supervisor agent's result. 
                If this is a dictionary, it should contain a 'department' field.

        Returns:
            str: The final department decision.
        """
        if isinstance(prediction_supervison, dict):
            department = prediction_supervison.pop('department')
            return department
        return prediction_department
        
    
    def _sanity_check(self,
                      prediction: dict, 
                      gt: dict,
                      conversations: str) -> Tuple[bool, str, Union[str, dict], dict]:
        """
        Performs a sanity check on the predicted patient information and department against the ground truth.

        This method validates:
        1. The format of the prediction (must be a dictionary for patient info).
        2. Completeness of the conversation simulation (all ground truth patient values must appear in the conversation).
        3. Accuracy of both the predicted department and patient information.

        Args:
            prediction (dict): The output generated by the model. Expected to contain:
                - 'patient': dict of patient demographic information (e.g., name, birth date, gender)
                - 'department': str representing the predicted department
            gt (dict): The ground truth data. Expected to contain:
                - 'patient': dict of correct patient demographic information
                - 'department': str of the correct department
            conversations (str): The full conversation text between the patient and administration staff.

        Returns:
            Tuple[bool, str, Union[str, dict], dict]:
                - bool: True if the prediction passes all sanity checks, False otherwise
                - str: Status code indicating the type of check passed or failed
                - Union[str, dict]: The prediction that was checked (or raw value if parsing failed)
                - dict: Additional details or metadata (currently returns the prediction dict)
        """
        ############################ Check the prediciton format #############################
        if not isinstance(prediction['patient'], dict):
            return False, STATUS_CODES['format'], prediction    # Could not be parsed as a dictionary
        
        ############################ Incomplete simulation case #############################
        # TODO: birth date가 simulation에 나왔는지 여부 판단하는 로직 추가 필요
        if not all(v in conversations for k, v in gt['patient'].items() if k != 'birth_date'):
            return False, STATUS_CODES['simulation'], prediction
        
        ############################ Check with the ground truth #############################
        wrong_department = prediction['department'][0] not in gt['department']
        wrong_info = prediction['patient'] != gt['patient']
        if wrong_department and wrong_info:
            return False, STATUS_CODES['department & patient'], prediction
        elif wrong_department:
            return False, STATUS_CODES['department'], prediction
        elif wrong_info:
            return False, STATUS_CODES['patient'], prediction
        
        return True, STATUS_CODES['correct'], prediction
        

    def __call__(self, data_pair: Tuple[dict, dict],  agent_test_data: dict, agent_results: dict, environment) -> dict:
        """
        Estimates the most appropriate medical department for each patient using an LLM agent.

        Args:
            data_pair (Tuple[dict, dict]): A pair of ground truth and patient data for agent simulation.
            agent_test_data (dict): A dictionary containing test data for a single hospital.
                Expected to include:
                    - 'department': Dictionary of available departments.
            agent_results (dict): Placeholder for compatibility; not used in this method.
            environment (HospitalEnvironment): Hospital environment instance to manage patient schedules.

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
        name, gender, birth_date = gt['patient'], gt['gender'], gt['birthDate']
        gt_data = {
            'patient': {
                'name': name, 
                'birth_date': birth_date, 
                'gender': gender
            }, 
            'department': gt['department']
        }
        results['gt'].append(gt_data)

        # LLM call: Conversation and department decision
        department_candidates = test_data['constraint']['symptom']['department']    # NOTE: Same as gt['department]
        diagnosis = 'Unknown for now' if test_data['constraint']['symptom_level'] == 'simple' else test_data['constraint']['symptom']['disease']   # NOTE: `simple` or `with_history`
        patient_agent = PatientAgent(
            self.task_model,
            'outpatient',
            lang_proficiency_level='B',
            department=department_candidates,
            name=name,
            birth_date=birth_date,
            gender=gender,
            diagnosis=diagnosis,
            chiefcomplaint=test_data['constraint']['symptom']['symptom'],
            random_seed=42,
            temperature=0
        )
        admin_staff_agent = AdminStaffAgent(
            self.task_model,
            departments,
            random_seed=42,
            temperature=0
        )
        environment = OPSimulation(patient_agent, admin_staff_agent)
        dialogs = environment.simulate(verbose=False)
        prediction_department = OutpatientIntake.postprocessing_department(dialogs[-1]['content'])

        # LLM call: Supervisor which should extract demographic information of the patient and evaluation the department decision result
        dialogs = '\n'.join([f"{turn['role']}: {' '.join(turn['content'].split())}" for turn in dialogs])
        if self.ensure_output_format:
            prediction_supervision = self.supervisor_client(
                user_prompt=self.user_prompt_template,
                system_prompt=self.system_prompt,
                **{
                    'CONVERSATION': dialogs,
                    'DEPARTMENTS': ''.join([f'{i+1}. {department}\n' for i, department in enumerate(departments)])
                }
            )
        else:
            user_prompt = self.user_prompt_template.format(
                CONVERSATION=dialogs,
                DEPARTMENTS=''.join([f'{i+1}. {department}\n' for i, department in enumerate(departments)])
            )
            prediction_supervision = self.supervisor_client(
                user_prompt,
                system_prompt=self.system_prompt, 
                using_multi_turn=False,
                verbose=False
            )
        prediction_supervision = OutpatientIntake.postprocessing_information(prediction_supervision)

        # Sanity check
        department = self._department_decision(prediction_department, prediction_supervision)
        prediction = {'patient': prediction_supervision, 'department': [department]}
        status, status_code, prediction = self._sanity_check(
            prediction=prediction,
            gt=gt_data,
            conversations=dialogs
        )
        
        # Append results
        results['pred'].append(prediction)
        results['status'].append(status)
        results['status_code'].append(status_code)
        
        return results



class AssignSchedule(Task):
    def __init__(self, config):
        # Initialize variables
        self.name = 'schedule'
        self.use_supervisor = config.schedule_task.use_supervisor
        self.task_model = config.task_model
        self._task_system_prompt_path = config.schedule_task.task_system_prompt
        self._task_user_prompt_path = config.schedule_task.task_user_prompt
        self.ensure_output_format = config.ensure_output_format
        self.integration_with_fhir = config.integration_with_fhir
        self.preference_phrase = {
            'asap': 'The patient wants the earliest available doctor in the department for the outpatient visit.',
            'doctor': 'The patient has a preferred doctor for the outpatient visit.'
        }

        # Initialize prompts
        self.task_system_prompt = txt_load(self._task_system_prompt_path)
        self.task_user_prompt_template = txt_load(self._task_user_prompt_path)

        # Initialize task model
        if 'gemini' in self.task_model.lower():
            self.task_client = GeminiLangChainClient(self.task_model) if self.ensure_output_format else GeminiClient(self.task_model)
        elif 'gpt' in self.task_model.lower():
            self.task_client = GPTLangChainClient(self.task_model) if self.ensure_output_format else GPTClient(self.task_model)
        else:
            self.task_client = VLLMClient(self.task_model, config.vllm_url)

        # If you use supervisor model
        if self.use_supervisor:
            self.supervisor_model = config.supervisor_model
            self._sup_system_prompt_path = config.schedule_task.supervisor_system_prompt
            self._sup_user_prompt_path = config.schedule_task.supervisor_user_prompt
            self.sup_system_prompt = txt_load(self._sup_system_prompt_path)
            self.sup_user_prompt_template = txt_load(self._sup_user_prompt_path)
            self.max_feedback_number = config.schedule_task.max_feedback_number

            if 'gemini' in self.supervisor_model.lower():
                self.supervisor_client = GeminiClient(self.supervisor_model)
            elif 'gpt' in self.supervisor_model.lower():
                self.supervisor_client = GPTClient(self.supervisor_model)
            else:
                self.supervisor_client = VLLMClient(self.supervisor_model, config.vllm_url)
        
    
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
                if not match:
                    return text
                json_str = match.group(1)
                text_dict = json.loads(json_str)
                
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


    def __extract_departments(self, gt: dict, agent_results: dict) -> Tuple[list[str], bool]:
        """
        Extracts the predicted department from agent results.
        If predictions are not available, falls back to using ground truth labels.

        Args:
            gt (dict): Ground truth data of a patient.
            agent_results (dict): A dictionary that may contain predicted department results under the key 'department'.

        Returns:
            Tuple[list[str], bool]: A department list, either predicted or ground truth and its sanity status.
        """
        try:
            department = agent_results['intake']['pred'][-1]['department'][0]
            sanity = agent_results['intake']['status'][-1]
        except:
            log('The predicted department is not given. The ground truth value will be used.', 'warning')
            department = gt['department'][0]
            sanity = True

        return department, sanity
    

    def __filter_doctor_schedule(self, doctor_information: dict, department: str, environment) -> dict:
        """
        Filter doctor information by department.

        Args:
            doctor_information (dict): A dictionary containing information about doctors, 
                                       including their department and schedule details.
            department (str): The department name used to filter the doctors.
            environment (Environment): Environment object containing booking number of each doctor.

        Returns:
            dict: A dictionary containing only the doctors who belong to the specified 
                  department, stored under the 'doctor' key.
        """
        filtered_doctor_information = {'doctor': {}}

        for k, v in doctor_information.items():
            if v['department'] == department:
                tmp_schedule = deepcopy(v)
                del tmp_schedule['capacity_per_hour'], tmp_schedule['capacity'], tmp_schedule['gender'], tmp_schedule['telecom'], tmp_schedule['birthDate']
                tmp_schedule['workload'] = f"{round(environment.booking_num[k] / v['capacity'] * 100, 2)}%"
                tmp_schedule['outpatient_duration'] = 1 / v['capacity_per_hour']
                filtered_doctor_information['doctor'][k] = tmp_schedule
        
        return filtered_doctor_information
    

    def __check_is_earlist(self, prediction: dict, doctor_information: dict, department: str, environment, preference_type: str) -> bool:
        """
        Check if the predicted schedule is the earliest possible option.

        Args:
            prediction (dict): Predicted schedule information including doctor, start time, end time, and date.
            doctor_information (dict): Dictionary containing doctors' schedules and availability.
            department (str): Department name used to filter relevant doctors.
            environment (Environment): Environment object containing current time and UTC offset.
            preference_type (str): Scheduling preference type, e.g., 'doctor' (specific doctor) or 'department'.

        Returns:
            bool: True if the predicted schedule is the earliest available, False otherwise.
        """
        fixed_schedules = self.__filter_doctor_schedule(doctor_information, department, environment)['doctor']
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
            
            fixed_schedule = v['schedule']
            for date, schedule in fixed_schedule.items():
                # date > pred_date case
                if compare_iso_time(date, pred_date):
                    continue

                fixed_schedule_segments = sum([convert_time_to_segment(self._START_HOUR, 
                                                                       self._END_HOUR, 
                                                                       self._TIME_UNIT, 
                                                                       fs) for fs in schedule], [])
                if date == pred_date:                
                    free_time = [s for s in range(prediction_schedule_segments[0]) if s not in fixed_schedule_segments]
                # date < pred_date case
                else:
                    all_time_segments = convert_time_to_segment(self._START_HOUR, self._END_HOUR, self._TIME_UNIT)
                    free_time = [s for s in range(len(all_time_segments)) if s not in fixed_schedule_segments]

                if len(free_time):
                    free_max_st, _ = convert_segment_to_time(self._START_HOUR, self._END_HOUR, self._TIME_UNIT, [free_time[-1]])
                    free_max_st_iso = get_iso_time(free_max_st, date, utc_offset=utc_offset)

                    if compare_iso_time(free_max_st_iso, current_time):
                        return False
        return True
    

    def _sanity_check(self,
                      prediction: Union[str, dict], 
                      patient_condition: dict,
                      doctor_information: dict,
                      environment) -> Tuple[bool, str, Union[str, dict], dict]:
        """
        Validates a predicted schedule for a doctor by checking its structure, time validity, 
        duplication with existing schedules, and updates the doctor's schedule if valid.

        Args:
            prediction (Union[str, dict]): The predicted allocation result, either a string (if parsing failed)
                                           or a dictionary mapping a doctor's name to a schedule with 'start' and 'end' times.
            patient_condition (dict): The conditions including name, preference, etc.
            doctor_information (dict): Dictionary of doctor data including their existing schedules.
                                       Each key is a doctor's name, and each value includes a 'schedule' field.
            environment (HospitalEnvironment): Hospital environment instance to manage patient schedules.

        Returns:
            Tuple[bool, str, Union[str, dict], dict]: 
                - A boolean indicating whether the prediction passed all sanity checks.
                - A string explaining its status.
                - The original prediction (wrong case) or processed prediction (correct case) (either unchanged or used for debugging/logging if invalid).
                - Updated doctor information after processing the prediction.
        """
        ############################ Check the prediciton format #############################
        if not isinstance(prediction, dict):
            return False, STATUS_CODES['format'], prediction, doctor_information    # Could not be parsed as a dictionary
        elif len(prediction['schedule']) > 1:
            return False, STATUS_CODES['conflict']['physician'], prediction, doctor_information    # Allocated more than one doctor; cannot determine target
    
        ################## Check the predicted schedule type and validities ##################
        try:
            doctor_name = list(prediction['schedule'].keys())[0]
            start = prediction['schedule'][doctor_name]['start']
            end = prediction['schedule'][doctor_name]['end']
            date = prediction['schedule'][doctor_name]['date']
            fixed_schedules = doctor_information[doctor_name]['schedule']
            start_iso_time = get_iso_time(start, date, utc_offset=environment._utc_offset)
            assert isinstance(start, float) and isinstance(end, float) and isinstance(date, str) \
                and start < end and start >= self._START_HOUR and end <= self._END_HOUR \
                and compare_iso_time(start_iso_time, environment.current_time) and date in fixed_schedules
            assert patient_condition['department'] == doctor_information[doctor_name]['department']
            
            # Duration mismatched case
            if not round(1 / doctor_information[doctor_name]['capacity_per_hour'], 4) == round(end - start, 4):
                return False, STATUS_CODES['duration'], prediction, doctor_information
            
        except KeyError:
            return False, STATUS_CODES['format'], prediction, doctor_information    # Schedule allocation missing or doctor not found
        except AssertionError:
            return False, STATUS_CODES['schedule'], prediction, doctor_information    # Invalid schedule times or department

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
            return False, STATUS_CODES['conflict']['time'], prediction, doctor_information    # Overlaps with an existing schedule
        
        ####################### Check the patient's preferences  #######################
        if patient_condition['preference'] == 'doctor':
            if patient_condition.get('preferred_doctor') != doctor_name:
                return False, STATUS_CODES['preference']['physician'], prediction, doctor_information
        
        is_earlist = self.__check_is_earlist(
            prediction,
            doctor_information,
            patient_condition['department'],
            environment,
            patient_condition['preference']
        )
        if not is_earlist:
            return False, STATUS_CODES['preference']['asap'], prediction, doctor_information
        
        ###################### Check the doctors' workload balance  #####################
        if not patient_condition['preference'] == 'doctor':
            schedule_candidates = self.__filter_doctor_schedule(
                doctor_information,
                patient_condition.get('department'),
                environment
            )['doctor']
            selected_doctor_wl = float(schedule_candidates[doctor_name]['workload'][:-1])
            if not all(float(v['workload'][:-1]) >= selected_doctor_wl for k, v in schedule_candidates.items() if k != doctor_name):
                return False, STATUS_CODES['workload'], prediction, doctor_information 
                
        # Finally update schedule of the doctor
        doctor_information[doctor_name]['schedule'][date].append([start, end])    # In-place logic
        doctor_information[doctor_name]['schedule'][date].sort()
        prediction = {
            'patient': patient_condition.get('patient'),
            'attending_physician': doctor_name,
            'department': patient_condition.get('department'),
            'date': date,
            'schedule': [start, end],
            'preference': patient_condition.get('preference'),
            'preferred_doctor': patient_condition.get('preferred_doctor'),
        }
        return True, STATUS_CODES['correct'], prediction, doctor_information
    

    def feedback(self, prediction: str, error_code: str, doctor_information: dict, environment) -> str:
        """
        Generate supervisor feedback based on scheduling results, error codes, and doctor information.

        Args:
            prediction (str): The scheduling result or predicted schedule to be evaluated.
            error_code (str): The code representing the type of scheduling error encountered.
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                       including availability and other relevant details.
            environment (Environment): Hospital environment.

        Returns:
            str: The feedback generated by the supervisor client, providing guidance on how to
                 correct the scheduling error.
        """
        user_prompt = self.sup_user_prompt_template.format(
            START_HOUR=self._START_HOUR,
            END_HOUR=self._END_HOUR,
            TIME_UNIT=self._TIME_UNIT,
            CURRENT_TIME=environment.current_time,
            DAY=self._DAY,
            DOCTOR=json.dumps(doctor_information, indent=2),
            RESULTS=prediction,
            ERROR_CODE=error_code,
            REASON='\n'.join(SCHEDULING_ERROR_CAUSE[error_code])
        )
        feedback = self.supervisor_client(
            user_prompt,
            system_prompt=self.sup_system_prompt, 
            using_multi_turn=False,
            verbose=False
        )
        return feedback
            

    def __call__(self, data_pair: Tuple[dict, dict], agent_test_data: dict, agent_results: dict, environment) -> dict:
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

        Returns:
            dict: A dictionary with three keys:
                - 'gt': List of ground truth results, each including patient info, attending physician, department, and schedule.
                - 'pred': List of predicted results (either valid dict or fallback string).
                - 'status': List of booleans indicating whether each prediction passed sanity checks.
                - 'status_code': List of status codes explaining each status.
        """
        gt, test_data = data_pair
        metadata = agent_test_data.get('metadata')
        department_data = agent_test_data.get('department')
        self._START_HOUR = metadata.get('time').get('start_hour')
        self._END_HOUR = metadata.get('time').get('end_hour')
        self._TIME_UNIT = metadata.get('time').get('interval_hour')
        self._DAY = metadata.get('days')
        doctor_information = environment.doctor_info_from_fhir() if self.integration_with_fhir else agent_test_data.get('doctor')
        department, sanity = self.__extract_departments(gt, agent_results)
        patient_condition = {
            'patient': test_data.get('patient'), 
            'department': department, 
            'preference': test_data.get('constraint').get('preference'),
            'preferred_doctor': test_data.get('constraint').get('attending_physician') if test_data.get('constraint').get('preference') == 'doctor' else "Doesn't matter",
            'symptom_level': test_data.get('constraint').get('symptom_level')
        }
        results = self.get_result_dict()

        # Append an example of exemplary answer
        gt_data = {
            'patient': gt.get('patient'),
            'attending_physician': gt.get('attending_physician'),
            'department': gt.get('department'),
            'preference': gt.get('preference'),
            'symptom_level': gt.get('symptom_level'),
        }
        results['gt'].append(gt_data)

        # If the precedent department data is wrong, continue
        if not sanity:
            results['pred'].append({})
            results['status'].append(False)
            results['status_code'].append(STATUS_CODES['preceding'])
            return results
        
        # LLM call and compare the validity of the LLM output
        feedback, prev_prediction = '', ''
        feedback_cnt = 0
        while 1:
            filtered_doctor_information = self.__filter_doctor_schedule(doctor_information, department, environment)
            if self.ensure_output_format:
                prediction = self.task_client(
                    user_prompt=self.task_user_prompt_template,
                    system_prompt=self.task_system_prompt,
                    **{
                        'START_HOUR': self._START_HOUR,
                        'END_HOUR': self._END_HOUR,
                        'TIME_UNIT': self._TIME_UNIT,
                        'CURRENT_TIME': environment.current_time,
                        'DEPARTMENT': department,
                        'PREFERENCE': self.preference_phrase[patient_condition.get('preference')],
                        'PREFERRED_DOCTOR': patient_condition.get('preferred_doctor'),
                        'DAY': self._DAY,
                        'DOCTOR': json.dumps(filtered_doctor_information, indent=2),
                        'PREV_ANSWER': prev_prediction,
                        'FEEDBACK': feedback,
                    }
                )

            else:
                user_prompt = self.task_user_prompt_template.format(
                    START_HOUR=self._START_HOUR,
                    END_HOUR=self._END_HOUR,
                    TIME_UNIT=self._TIME_UNIT,
                    CURRENT_TIME=environment.current_time,
                    DEPARTMENT=department,
                    PREFERENCE=self.preference_phrase[patient_condition.get('preference')],
                    PREFERRED_DOCTOR=patient_condition.get('preferred_doctor'),
                    DAY=self._DAY,
                    DOCTOR=json.dumps(filtered_doctor_information, indent=2),
                    PREV_ANSWER=prev_prediction,
                    FEEDBACK= feedback,
                )
                prediction = self.task_client(
                    user_prompt,
                    system_prompt=self.task_system_prompt, 
                    using_multi_turn=self.use_supervisor,
                    verbose=False
                )
            prediction = AssignSchedule.postprocessing(prediction)    
            status, status_code, prediction, doctor_information = self._sanity_check(
                prediction, 
                patient_condition,
                doctor_information,
                environment
            )
            
            if not status and self.use_supervisor and feedback_cnt < self.max_feedback_number:
                prev_prediction += json.dumps(prediction) + f': {status_code}\n'
                feedback = self.feedback(prediction, status_code, filtered_doctor_information, environment)
                feedback_cnt += 1
                continue
            
            break

        # POST/PUT to FHIR
        fhir_patient, fhir_appointment = None, None
        if status and self.integration_with_fhir:
            # Even if a failure occurs during a later API tasks, update the FHIR resources to ensure continued scheduling task 
            fhir_patient = DataConverter.data_to_patient(
                {'metadata': deepcopy(metadata),
                 'department': deepcopy(department_data),
                 'patient': {test_data['patient']: {'department': department, **deepcopy(test_data)}}}
            )[0]
            fhir_appointment = self._get_fhir_appointment(data={'metadata': deepcopy(metadata),
                                                                'department': deepcopy(department_data),
                                                                'information': deepcopy(prediction)})
            
        agent_test_data['doctor'] = doctor_information    # Update the doctor information in the agent test data
        environment.update_env(
            status=status, 
            patient_schedule=prediction,
            fhir_resources={'Patient': fhir_patient, 'Appointment': fhir_appointment}
        )
        
        # Append results
        results['pred'].append(prediction)
        results['status'].append(status)
        results['status_code'].append(status_code)
        
        return results



# class MakeFHIRResource(Task):
#     def __init__(self, config):
#         self.name = 'fhir_resource'
#         self.__init_env(config)
#         self.system_prompt = txt_load(self._system_prompt_path)
#         self.user_prompt_template = txt_load(self._user_prompt_path)
#         if 'gemini' in config.model.lower():
#             self.client = GeminiClient(config.model)
#         elif 'gpt' in config.model.lower():
#             self.client = GPTClient(config.model)
#         else:
#             self.client = VLLMClient(config.model, config.vllm_url)


#     def __init_env(self, config):
#         """
#         Initialize necessary variables.

#         Args:
#             config (Config): Configuration for agent tasks.
#         """
#         self._system_prompt_path = config.fhir_resource_task.system_prompt
#         self._user_prompt_path = config.fhir_resource_task.user_prompt
#         self._fhir_data_path = Path(config.fhir_data)


#     @staticmethod
#     def postprocessing(text: str) -> Union[str, dict]:
#         """
#         Attempts to parse the given text as JSON. If parsing succeeds, returns a dictionary;
#         otherwise, returns the original string.

#         Args:
#             text (str): The text output to post-process, potentially a JSON-formatted string.

#         Returns:
#             Union[str, dict]: A dictionary if the text is valid JSON, otherwise the original string.
#         """
#         try:
#             match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
#             if match:
#                 json_str = match.group(1)
#                 return json.loads(json_str)
#             return text
#         except:
#             return text
            
    
#     def __extract_schedules(self, gt: Tuple[dict, dict], agent_results: dict) -> Tuple[dict, bool]:
#         """
#         Extracts the predicted schedule from agent results.
#         If predictions are not available, falls back to using ground truth labels.

#         Args:
#             gt (dict): Ground truth data of a patient.
#             agent_results (dict): A dictionary that may contain predicted schedule results under the key 'schedule'.

#         Returns:
#             Tuple[dict, bool]: A scedule dictionary, either predicted or ground truth and its sanity status.
#         """
#         try:
#             schedule = agent_results['schedule']['pred'][-1]
#             sanity = agent_results['schedule']['status'][-1]
#         except:
#             log('The predicted schedule is not given. The ground truth value will be used.', 'warning')
#             schedule = {
#                 'patient': gt['patient'], 
#                 'attending_physician': gt['attending_physician'],
#                 'department': gt['department'],
#                 'schedule': gt['schedule']['time'],
#                 'priority': gt['priority'],
#                 'flexibility': gt['flexibility']
#             }
#             sanity = True

#         return schedule, sanity
    

#     def _sanity_check(self,
#                       prediction: Union[str, dict], 
#                       expected_prediction: dict) -> Tuple[bool, str, Union[str, dict]]:
#         """
#         Perform a recursive sanity check to compare the predicted Appointment resource against the expected one.

#         Args:
#             prediction (Union[str, dict]): The predicted Appointment resource, either as 
#                                            a JSON-parsed dictionary or a raw string (which will be rejected as invalid).
#             expected_prediction (dict): The expected Appointment resource structure to validate against.

#         Returns:
#             Tuple[bool, str, Union[str, dict]]:
#                 - A boolean indicating whether the prediction passed the sanity check.
#                 - A status code string (e.g., 'correct', 'format') from `STATUS_CODES`.
#                 - The original prediction, for logging or further processing.
#         """
#         # Check the prediciton format
#         if not isinstance(prediction, dict):
#             return False, STATUS_CODES['format'], prediction    # Could not be parsed as a dictionary
        
#         # Compare recursively
#         if self.sanity_check_recursively(expected_prediction, prediction):
#             return True, STATUS_CODES['correct'], prediction
#         else:
#             return False, STATUS_CODES['format'], prediction
    

#     def __get_necessary_information(self, schedule: dict) -> dict:
#         """
#         Extract and generate necessary information for creating a FHIR Appointment resource.

#         This includes identifiers and references for the patient and practitioner, ISO-formatted
#         start and end times, slot references based on schedule segments, and appointment ID.

#         Args:
#             schedule (dict): A dictionary containing patient appointment data.
#                             Expected keys include: 'patient', 'attending_physician',
#                             'department', and 'schedule'.

#         Returns:
#             dict: A dictionary containing the following fields:
#                 - 'ID' (str): Unique appointment ID based on practitioner and time.
#                 - 'START_HOUR' (str): ISO 8601 formatted start time.
#                 - 'END_HOUR' (str): ISO 8601 formatted end time.
#                 - 'SLOT' (list[str]): List of Slot resource references.
#                 - 'PATIENT' (str): Patient name or identifier.
#                 - 'PATIENT_REF' (str): FHIR reference string to the Patient resource.
#                 - 'PRACTITIONER' (str): Doctor name or identifier.
#                 - 'PRACTITIONER_REF' (str): FHIR reference string to the Practitioner resource.
#         """
#         patient = schedule['patient']
#         doctor = schedule['attending_physician']
#         department = schedule['department']
#         date = schedule['date']
#         schedule = schedule['schedule']
        
#         utc_offset = get_utc_offset(self._country_code)
#         patient_id = get_individual_id(self._hospital_name, self._department_data[department]['code'], patient)
#         practitioner_id = get_individual_id(self._hospital_name, self._department_data[department]['code'], doctor)
#         schedule_segments = convert_time_to_segment(self._START_HOUR, self._END_HOUR, self._TIME_UNIT, schedule)
        
#         _id = get_appointment_id(practitioner_id, date, schedule_segments[0], schedule_segments[-1])
#         start = get_iso_time(schedule[0], utc_offset=utc_offset)
#         end = get_iso_time(schedule[-1], utc_offset=utc_offset)
#         patient_ref = f'Patient/{patient_id}'
#         practitioner_ref = f'Practitioner/{practitioner_id}'
#         slot = [f'Slot/{practitioner_id}-slot{seg}' for seg in schedule_segments]

#         return {
#             'ID': _id,
#             'START_HOUR': start,
#             'END_HOUR': end,
#             'SLOT': slot,
#             'PATIENT': patient,
#             'PATIENT_REF': patient_ref,
#             'PRACTITIONER': doctor,
#             'PRACTITIONER_REF': practitioner_ref,
#         }


#     def __call__(self, data_pair: Tuple[dict, dict], agent_test_data: dict, agent_results: dict, environment) -> dict:
#         """
#         Run the evaluation pipeline for generating and validating FHIR Appointment resources based on agent scheduling results.

#         Args:
#             data_pair (Tuple[dict, dict]): A pair of ground truth and patient data for agent simulation.
#             agent_test_data (dict): Dictionary containing test metadata, doctor/patient/schedule data,
#                                     and ground-truth information for each case.
#             agent_results (dict): Dictionary containing scheduling results generated by the agent.
#             environment (HospitalEnvironment): Hospital environment instance to manage patient schedules.

#         Returns:
#             dict: A dictionary with the following keys:
#                 - 'gt': List of ground-truth FHIR Appointment resources.
#                 - 'pred': List of predicted FHIR Appointment resources generated by the LLM.
#                 - 'status': List of boolean values indicating if each prediction passed the sanity check.
#                 - 'status_code': List of string codes representing the validation status.
#         """
#         gt, test_data = data_pair
#         self._department_data = agent_test_data.get('department')
#         metadata = agent_test_data.get('metadata')
#         self._hospital_name = metadata.get('hospital_name')
#         self._country_code = metadata.get('country_code', 'KR')
#         self._START_HOUR = metadata.get('time').get('start_hour')
#         self._END_HOUR = metadata.get('time').get('end_hour')
#         self._TIME_UNIT = metadata.get('time').get('interval_hour')
#         schedule, sanity = self.__extract_schedules(gt, agent_results)
#         results = self.get_result_dict()
        
#         # Load ground truth Appointment FHIR resource
#         gt_resource = self._get_fhir_appointment(self._fhir_data_path / 'appointment' / gt.get('fhir_resource'))
#         results['gt'].append(gt_resource)

#         # If the precedent schedule data is invalid, continue
#         if not sanity:
#             results['pred'].append({})
#             results['status'].append(False)
#             results['status_code'].append(STATUS_CODES['preceding'])
#             return results

#         # LLM call and compare the validity of the LLM output
#         user_prompt = self.user_prompt_template.format(**self.__get_necessary_information(schedule))
#         prediction = self.client(
#             user_prompt,
#             system_prompt=self.system_prompt, 
#             using_multi_turn=False
#         )
#         prediction = MakeFHIRResource.postprocessing(prediction)
#         expected_prediction = self._get_fhir_appointment(data={'metadata': deepcopy(metadata), 
#                                                                'department': deepcopy(self._department_data), 
#                                                                'information': deepcopy(schedule)})
#         status, status_code, prediction = self._sanity_check(prediction, expected_prediction)
        
#         # Append results
#         results['pred'].append(prediction)
#         results['status'].append(status)
#         results['status_code'].append(status_code)
            
#         return results



# class MakeFHIRAPI(Task):
#     def __init__(self, config):
#         self.name = 'fhir_api'
#         self.__init_env(config)
#         self.system_prompt = txt_load(self._system_prompt_path)
#         self.user_prompt_template = txt_load(self._user_prompt_path)
#         if 'gemini' in config.model.lower():
#             self.client = GeminiClient(config.model)
#         elif 'gpt' in config.model.lower():
#             self.client = GPTClient(config.model)
#         else:
#             self.client = VLLMClient(config.model, config.vllm_url)


#     def __init_env(self, config):
#         """
#         Initialize necessary variables.

#         Args:
#             config (Config): Configuration for agent tasks.
#         """
#         self._system_prompt_path = config.fhir_api_task.system_prompt
#         self._user_prompt_path = config.fhir_api_task.user_prompt
#         self._fhir_data_path = Path(config.fhir_data)
#         self._fhir_url = config.fhir_url


#     @staticmethod
#     def postprocessing(text: str) -> Union[str, None]:
#         """
#         Extracts a Bash command from a markdown-formatted code block in the given text.

#         Args:
#             text (str): The input text potentially containing a markdown-formatted Bash code block.

#         Returns:
#             Union[str, None]: The extracted Bash command if found, otherwise None.
#         """
#         try:
#             match = re.search(r'```bash\s*(.*?)\s*```', text, re.DOTALL)
#             if match:
#                 return match.group(1)
#         except:
#             return None
        

#     def __extract_fhir_resource(self, gt: dict, agent_results: dict) -> Tuple[list[dict], list[bool]]:
#         """
#         Extracts the predicted FHIR Appointment resource from agent results.
#         If predictions are not available, falls back to using ground truth labels.

#         Args:
#             data_pair (Tuple[dict, dict]): A pair of (ground_truth, test_data) for a single patient.
#             gt (dict): Ground truth data of a patient.

#         Returns:
#             Tuple[list[dict], list[bool]]: A list of fhir_resources, either predicted or ground truth and each sanity status.
#         """
#         try:
#             fhir_resource = agent_results['fhir_resource']['pred'][-1]
#             sanity = agent_results['fhir_resource']['status'][-1]
#         except:
#             log('The predicted fhir_resource is not given. The ground truth value will be used.', 'warning')
#             fhir_resource_path = self._fhir_data_path / 'appointment' / gt.get('fhir_resource')
            
#             # Load the FHIR json file if it exists, otherwise create a new one
#             fhir_resource = self._get_fhir_appointment(
#                 gt_resource_path=fhir_resource_path,
#                 data={'metadata': deepcopy(self._metadata), 
#                       'department': deepcopy(self._department_data),
#                       'information': deepcopy(gt)}
#             )
#             sanity = True
        

#         return fhir_resource, sanity
    

#     def _sanity_check(self,
#                       prediction: Union[str, None], 
#                       expected_prediction: str) -> Tuple[bool, str, Union[str, None]]:
#         """
#         Perform a sanity check to compare the predicted curl command against the expected one.

#         Args:
#             prediction (Union[str, None]): The LLM-generated curl command.
#             expected_prediction (str): The expected curl command.

#         Returns:
#             Tuple[bool, str, Union[str, None]]:
#                 - A boolean indicating whether the prediction passed the sanity check.
#                 - A status code string (e.g., 'correct', 'format') from `STATUS_CODES`.
#                 - The original prediction, for logging or further processing.
#         """
#         def check_curl(curl_command: str) -> bool:
#             return curl_command.startswith('curl -X PUT')

#         def extract_url(curl_command: str) -> str:
#             match = re.search(r'(http[s]?://[^\s"]+)', curl_command)
#             return match.group(0).strip() if match else ''

#         def extract_headers(curl_command: str) -> set:
#             pattern = r'-(?:H|-header) "([^"]+)"'
#             return set(re.findall(pattern, curl_command))
        
#         def extract_payload(curl_command: str) -> dict:
#             pattern = r"-(?:d|-data)\s+(['\"])(\{.*?\})\1"  # ' 또는 "로 감싼 JSON 추출
#             match = re.search(pattern, curl_command, re.DOTALL)
#             if match:
#                 try:
#                     return json.loads(match.group(2))
#                 except json.JSONDecodeError:
#                     pass
#             return {}
        
#         # Check the prediciton format
#         if not isinstance(prediction, str):
#             return False, STATUS_CODES['format'], prediction    # Could not be parsed as a dictionary

#         # Check the sanity of the generated command
#         if not check_curl(prediction):
#             return False, STATUS_CODES['curl']['http'], prediction
#         if not extract_url(prediction) == extract_url(expected_prediction):
#             return False, STATUS_CODES['curl']['url'], prediction
#         if not extract_headers(prediction) == extract_headers(expected_prediction):
#             return False, STATUS_CODES['curl']['header'], prediction
#         if not self.sanity_check_recursively(extract_payload(prediction), extract_payload(expected_prediction)):
#             return False, STATUS_CODES['curl']['payload'], prediction
        
#         return True, STATUS_CODES['correct'], prediction
    

#     def __get_api(self, resource: dict) -> str:
#         """
#         Generate curl API command from a FHIR resource.

#         Args:
#             resource (dict): FHIR resource ditionary.

#         Returns:
#             str: The generated curl API command.
#         """
#         gt_api = f'curl -X PUT "{self._fhir_url}/Appointment/{resource["id"]}" ' \
#                  f'-H "Content-Type: application/fhir+json" ' \
#                  f'-H "Accept: application/fhir+json" ' \
#                  f"-d '{json.dumps(resource)}'"
#         return gt_api


#     def __call__(self, data_pair: Tuple[dict, dict], agent_test_data: dict, agent_results: dict, environment) -> dict:
#         """
#         Processes test data and LLM results to generate and validate FHIR Appointment resources.
        
#         Args:
#             data_pair (Tuple[dict, dict]): A pair of ground truth and patient data for agent simulation.
#             agent_test_data (dict): Input test data containing metadata and agent-specific data for processing.
#             agent_results (dict): Results produced by the agent (e.g., LLM output) to be validated.
#             environment (HospitalEnvironment): Hospital environment instance to manage patient schedules.

#         Returns:
#             dict: A dictionary containing lists for each processed item with keys:
#                 - 'gt': ground truth FHIR resource API commands.
#                 - 'pred': predicted FHIR resource API commands from the LLM.
#                 - 'status': boolean flags indicating pass/fail of sanity checks.
#                 - 'status_code': human-readable status codes explaining the validation result.
#         """
#         gt, test_data = data_pair
#         self._metadata = agent_test_data.get('metadata')
#         self._department_data = agent_test_data.get('department')
#         fhir_resource, sanity = self.__extract_fhir_resource(gt, agent_results)
#         results = self.get_result_dict()
        
#         # Append a ground truth Appointment FHIR resource and API command
#         gt_api = self.__get_api(
#             self._get_fhir_appointment(
#                 gt_resource_path=self._fhir_data_path / 'appointment' / gt.get('fhir_resource'),
#                 data={'metadata': deepcopy(self._metadata),
#                       'department': deepcopy(self._department_data),
#                       'information': deepcopy(gt)}
#             )
#         )
#         results['gt'].append(gt_api)

#         # If the precedent resource data is invalid, continue
#         if not sanity:
#             results['pred'].append('')
#             results['status'].append(False)
#             results['status_code'].append(STATUS_CODES['preceding'])
#             return results

#         # LLM call and compare the validity of the LLM output
#         user_prompt = self.user_prompt_template.format(FHIR_URL=self._fhir_url, RESOURCE=json.dumps(fhir_resource))
#         prediction = self.client(
#             user_prompt,
#             system_prompt=self.system_prompt, 
#             using_multi_turn=False
#         )
#         prediction = MakeFHIRAPI.postprocessing(prediction)
#         expected_prediction = self.__get_api(fhir_resource)
#         status, status_code, prediction = self._sanity_check(prediction, expected_prediction)
        
#         # Append results
#         results['pred'].append(prediction)
#         results['status'].append(status)
#         results['status_code'].append(status_code)
            
#         return results
