import json
from pathlib import Path
from copy import deepcopy
from typing import Tuple, Union, Optional

from tools import (
    GeminiClient,
    GeminiLangChainClient,
    GPTClient,
    GPTLangChainClient,
    DataConverter,
)
from utils import log
from utils.fhir_utils import *
from utils.filesys_utils import txt_load, json_load
from utils.common_utils import (
    convert_time_to_segment,
    get_utc_offset,
    get_iso_time,
)



class Task:
    # Definition of status codes
    status_codes = {
        'department': 'incorrect department',
        'format': 'incorrect format',
        'schedule': 'invalid schedule',
        'conflict': {'physician': 'physician conflict', 'time': 'time conflict'},
        'curl': {'http': 'incorrect http method', 'header': 'incorrect header', 'payload': 'incorrect payload', 'url': 'incorrect url'},
        'preceding': 'preceding task failed',
        'correct': 'pass',
    }


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
            metadata, info = data.get('metadata'), data.get('information')
            schedule = info.get('schedule')
            if 'time' in schedule:
                schedule = schedule.get('time')
            
            gt_resource = DataConverter.data_to_appointment(
                {
                    'metadata': metadata,
                    'patient': {
                        info.get('patient'): {
                            'department': info.get('department'),
                            'attending_physician': info.get('attending_physician'),
                            'schedule': schedule
                        }
                    }
                }
            )[0]
            return gt_resource
        
    
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



class AssignDepartment(Task):
    def __init__(self, config):
        self.name = 'department'
        self.__init_env(config)
        self.system_prompt = txt_load(self._system_prompt_path)
        self.user_prompt_template = txt_load(self._user_prompt_path)
        self.client = GeminiClient(config.model) if 'gemini' in config.model.lower() else GPTClient(config.model)


    def __init_env(self, config):
        """
        Initialize necessary variables.

        Args:
            config (Config): Configuration for agent tasks.
        """
        self._system_prompt_path = config.department_task.system_prompt
        self._user_prompt_path = config.department_task.user_prompt

    
    @staticmethod
    def postprocessing(text: str) -> str:
        """
        Post-processing method of text output.

        Args:
            text (str): Text input.

        Returns:
            str: Post-processed text output.
        """
        text = text.split()[-1].strip()
        return text


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
        options = ''.join([f'{i+1}. {department}\n' for i, department in enumerate(departments)])
        results = self.get_result_dict()
        
        # Append a ground truth
        gt_department = gt['department']
        results['gt'].append(gt_department)
        
        # LLM call
        user_prompt = self.user_prompt_template.format(SYMPTOM=test_data['symptom'], OPTIONS=options)
        prediction = self.client(
            user_prompt,
            system_prompt=self.system_prompt, 
            using_multi_turn=False
        )
        prediction = AssignDepartment.postprocessing(prediction)
        
        # Append results
        status = gt_department == prediction
        status_code = self.status_codes['correct'] if status else self.status_codes['department']
        results['pred'].append(prediction)
        results['status'].append(status)
        results['status_code'].append(status_code)
        
        return results



class AssignSchedule(Task):
    def __init__(self, config):
        self.name = 'schedule'
        self.__init_env(config)
        self.system_prompt = txt_load(self._system_prompt_path)
        self.user_prompt_template = txt_load(self._user_prompt_path)
        if self.ensure_output_format:
            self.client = GeminiLangChainClient(config.model) if 'gemini' in config.model.lower() else GPTLangChainClient(config.model)
        else:
            self.client = GeminiClient(config.model) if 'gemini' in config.model.lower() else GPTClient(config.model)


    def __init_env(self, config):
        """
        Initialize necessary variables.

        Args:
            config (Config): Configuration for agent tasks.
        """
        self._system_prompt_path = config.schedule_task.system_prompt
        self._user_prompt_path = config.schedule_task.user_prompt
        self.ensure_output_format = config.ensure_output_format
        self.integration_with_fhir = config.integration_with_fhir

    
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
            
            assert len(text_dict) == 2 and all(k in text_dict for k in ['schedule', 'changed_existing_schedule_list'])   # Basic sanity check
            key = list(text_dict['schedule'].keys())[0]
            text_dict['schedule'][key]['start'] = float(text_dict['schedule'][key]['start'])
            text_dict['schedule'][key]['end'] = float(text_dict['schedule'][key]['end'])
            return text_dict
        
        except:
            return str(text)


    def __extract_departments(self, gt: dict, agent_results: dict) -> Tuple[str, bool]:
        """
        Extracts the predicted department from agent results.
        If predictions are not available, falls back to using ground truth labels.

        Args:
            gt (dict): Ground truth data of a patient.
            agent_results (dict): A dictionary that may contain predicted department results under the key 'department'.

        Returns:
            Tuple[str, bool]: A department, either predicted or ground truth and its sanity status.
        """
        try:
            department = agent_results['department']['pred'][-1]
            sanity = agent_results['department']['status'][-1]
        except:
            log('The predicted department is not given. The ground truth value will be used.', 'warning')
            department = gt['department']
            sanity = True

        return department, sanity
    

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
            patient_condition (dict): The conditions including duration, etc.
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
            return False, self.status_codes['format'], prediction, doctor_information    # Could not be parsed as a dictionary
        else:
            if len(prediction['schedule']) > 1:
                return False, self.status_codes['conflict']['physician'], prediction, doctor_information    # Allocated more than one doctor; cannot determine target
            if not isinstance(prediction['changed_existing_schedule_list'], list):
                return False, self.status_codes['format'], prediction, doctor_information
            if not (len(prediction['changed_existing_schedule_list']) == 0 or all(isinstance(s, dict) for s in prediction['changed_existing_schedule_list'])):
                return False, self.status_codes['format'], prediction, doctor_information

        
        ################## Check the predicted changed schedules validities ##################
        sanity, status_code, tmp_doctor_information = environment._reschedule_sanity_check(
            prediction['changed_existing_schedule_list'],
            doctor_information,
            patient_condition,
        )
        if not sanity:
            return sanity, status_code, prediction, doctor_information

    
        ################## Check the predicted schedule type and validities ##################
        try:
            doctor_name = list(prediction['schedule'].keys())[0]
            start = prediction['schedule'][doctor_name]['start']
            end = prediction['schedule'][doctor_name]['end']
            fixed_schedules = tmp_doctor_information[doctor_name]['schedule']
            start_iso_time = get_iso_time(start, utc_offset=environment._utc_offset)
            assert isinstance(start, float) and isinstance(end, float) \
                and start < end and start >= self._START_HOUR and end <= self._END_HOUR and start_iso_time > environment.current_time
            assert patient_condition['department'] == tmp_doctor_information[doctor_name]['department'] \
                and patient_condition['duration'] == round(end - start, 4)
        except KeyError:
            return False, self.status_codes['format'], prediction, doctor_information    # Schedule allocation missing or doctor not found
        except AssertionError:
            return False, self.status_codes['schedule'], prediction, doctor_information    # Invalid schedule times or department

        ####################### Check the duplication of the schedules #######################
        prediction_schedule_segments = convert_time_to_segment(self._START_HOUR,
                                                               self._END_HOUR,
                                                               self._TIME_UNIT,
                                                               [start, end])
        fixed_schedule_segments = sum([convert_time_to_segment(self._START_HOUR, 
                                                               self._END_HOUR, 
                                                               self._TIME_UNIT, 
                                                               fs) for fs in fixed_schedules], [])
        
        if len(set(prediction_schedule_segments) & set(fixed_schedule_segments)):
            return False, self.status_codes['conflict']['time'], prediction, doctor_information    # Overlaps with an existing schedule
        
        # Finally update schedule of the doctor
        tmp_doctor_information[doctor_name]['schedule'].append([start, end])    # In-place logic
        prediction = {
            'patient': patient_condition.get('patient'),
            'attending_physician': doctor_name,
            'department': patient_condition.get('department'),
            'schedule': [start, end],
            'priority': patient_condition.get('priority'),
            'flexibility': patient_condition.get('flexibility'),
        }
        return True, self.status_codes['correct'], prediction, deepcopy(tmp_doctor_information)
            

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
        self._START_HOUR = metadata.get('time').get('start_hour')
        self._END_HOUR = metadata.get('time').get('end_hour')
        self._TIME_UNIT = metadata.get('time').get('interval_hour')
        doctor_information = environment.doctor_info_from_fhir() if self.integration_with_fhir else agent_test_data.get('doctor')
        department, sanity = self.__extract_departments(gt, agent_results)
        patient_condition = {
            'patient': test_data.get('patient'), 
            'department': department, 
            'duration': test_data.get('constraint').get('duration'), 
            'priority': test_data.get('constraint').get('priority'), 
            'flexibility': test_data.get('constraint').get('flexibility')
        }
        results = self.get_result_dict()

        # Append an example of exemplary answer
        gt_results = {
            'patient': gt.get('patient'),
            'attending_physician': gt.get('attending_physician'),
            'department': gt.get('department'),
            'schedule': gt.get('schedule').get('time'),
            'priority': gt.get('priority'),
            'flexibility': gt.get('flexibility')
        }
        results['gt'].append(gt_results)

        # If the precedent department data is wrong, continue
        if not sanity:
            results['pred'].append({})
            results['status'].append(False)
            results['status_code'].append(self.status_codes['preceding'])
            return results
        
        # LLM call and compare the validity of the LLM output
        if self.ensure_output_format:
            prediction = self.client(
                user_prompt=self.user_prompt_template,
                system_prompt=self.system_prompt,
                **{
                    'START_HOUR': self._START_HOUR,
                    'END_HOUR': self._END_HOUR,
                    'TIME_UNIT': self._TIME_UNIT,
                    'CURRENT_TIME': environment.current_time,
                    'DEPARTMENT': department,
                    'DURATION': patient_condition.get('duration'),
                    'PRIORITY': patient_condition.get('priority'),
                    'FLEXIBILITY': patient_condition.get('flexibility'),
                    'DOCTOR': json.dumps(doctor_information, indent=2),
                    'PATIENT_SCHEDULES': json.dumps(environment.patient_schedules, indent=2)
                }
            )

        else:
            user_prompt = self.user_prompt_template.format(
                START_HOUR=self._START_HOUR,
                END_HOUR=self._END_HOUR,
                TIME_UNIT=self._TIME_UNIT,
                CURRENT_TIME=environment.current_time,
                DEPARTMENT=department,
                DURATION=patient_condition.get('duration'),
                PRIORITY=patient_condition.get('priority'),
                FLEXIBILITY=patient_condition.get('flexibility'),
                DOCTOR=json.dumps(doctor_information, indent=2),
                PATIENT_SCHEDULES=json.dumps(environment.patient_schedules, indent=2)
            )
            prediction = self.client(
                user_prompt,
                system_prompt=self.system_prompt, 
                using_multi_turn=False
            )
        prediction = AssignSchedule.postprocessing(prediction)    
        status, status_code, prediction, doctor_information = self._sanity_check(
            prediction, 
            patient_condition,
            doctor_information,
            environment
        )

        # POST/PUT to FHIR
        fhir_patient, fhir_appointment = None, None
        if status and self.integration_with_fhir:
            # Even if a failure occurs during a later API tasks, update the FHIR resources to ensure continued scheduling task 
            fhir_patient = DataConverter.data_to_patient(
                {'metadata': deepcopy(metadata), 'patient': {test_data['patient']: {'department': department, **deepcopy(test_data)}}}
            )[0]
            fhir_appointment = self._get_fhir_appointment(data={'metadata': deepcopy(metadata),
                                                                'information': {'department': prediction.get('department'),
                                                                                'patient': prediction.get('patient'),
                                                                                'attending_physician': prediction.get('attending_physician'),
                                                                                'schedule': prediction.get('schedule')}})

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



class MakeFHIRResource(Task):
    def __init__(self, config):
        self.name = 'fhir_resource'
        self.__init_env(config)
        self.system_prompt = txt_load(self._system_prompt_path)
        self.user_prompt_template = txt_load(self._user_prompt_path)
        self.client = GeminiClient(config.model) if 'gemini' in config.model.lower() else GPTClient(config.model)


    def __init_env(self, config):
        """
        Initialize necessary variables.

        Args:
            config (Config): Configuration for agent tasks.
        """
        self._system_prompt_path = config.fhir_resource_task.system_prompt
        self._user_prompt_path = config.fhir_resource_task.user_prompt
        self._fhir_data_path = Path(config.fhir_data)


    @staticmethod
    def postprocessing(text: str) -> Union[str, dict]:
        """
        Attempts to parse the given text as JSON. If parsing succeeds, returns a dictionary;
        otherwise, returns the original string.

        Args:
            text (str): The text output to post-process, potentially a JSON-formatted string.

        Returns:
            Union[str, dict]: A dictionary if the text is valid JSON, otherwise the original string.
        """
        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            return text
        except:
            return text
            
    
    def __extract_schedules(self, gt: Tuple[dict, dict], agent_results: dict) -> Tuple[dict, bool]:
        """
        Extracts the predicted schedule from agent results.
        If predictions are not available, falls back to using ground truth labels.

        Args:
            gt (dict): Ground truth data of a patient.
            agent_results (dict): A dictionary that may contain predicted schedule results under the key 'schedule'.

        Returns:
            Tuple[dict, bool]: A scedule dictionary, either predicted or ground truth and its sanity status.
        """
        try:
            schedule = agent_results['schedule']['pred'][-1]
            sanity = agent_results['schedule']['status'][-1]
        except:
            log('The predicted schedule is not given. The ground truth value will be used.', 'warning')
            schedule = {
                'patient': gt['patient'], 
                'attending_physician': gt['attending_physician'],
                'department': gt['department'],
                'schedule': gt['schedule']['time'],
                'priority': gt['priority'],
                'flexibility': gt['flexibility']
            }
            sanity = True

        return schedule, sanity
    

    def _sanity_check(self,
                      prediction: Union[str, dict], 
                      expected_prediction: dict) -> Tuple[bool, str, Union[str, dict]]:
        """
        Perform a recursive sanity check to compare the predicted Appointment resource against the expected one.

        Args:
            prediction (Union[str, dict]): The predicted Appointment resource, either as 
                                           a JSON-parsed dictionary or a raw string (which will be rejected as invalid).
            expected_prediction (dict): The expected Appointment resource structure to validate against.

        Returns:
            Tuple[bool, str, Union[str, dict]]:
                - A boolean indicating whether the prediction passed the sanity check.
                - A status code string (e.g., 'correct', 'format') from `self.status_codes`.
                - The original prediction, for logging or further processing.
        """
        # Check the prediciton format
        if not isinstance(prediction, dict):
            return False, self.status_codes['format'], prediction    # Could not be parsed as a dictionary
        
        # Compare recursively
        if self.sanity_check_recursively(expected_prediction, prediction):
            return True, self.status_codes['correct'], prediction
        else:
            return False, self.status_codes['format'], prediction
    

    def __get_necessary_information(self, schedule: dict) -> dict:
        """
        Extract and generate necessary information for creating a FHIR Appointment resource.

        This includes identifiers and references for the patient and practitioner, ISO-formatted
        start and end times, slot references based on schedule segments, and appointment ID.

        Args:
            schedule (dict): A dictionary containing patient appointment data.
                            Expected keys include: 'patient', 'attending_physician',
                            'department', and 'schedule'.

        Returns:
            dict: A dictionary containing the following fields:
                - 'ID' (str): Unique appointment ID based on practitioner and time.
                - 'START_HOUR' (str): ISO 8601 formatted start time.
                - 'END_HOUR' (str): ISO 8601 formatted end time.
                - 'SLOT' (list[str]): List of Slot resource references.
                - 'PATIENT' (str): Patient name or identifier.
                - 'PATIENT_REF' (str): FHIR reference string to the Patient resource.
                - 'PRACTITIONER' (str): Doctor name or identifier.
                - 'PRACTITIONER_REF' (str): FHIR reference string to the Practitioner resource.
        """
        patient = schedule['patient']
        doctor = schedule['attending_physician']
        department = schedule['department']
        schedule = schedule['schedule']
        
        utc_offset = get_utc_offset(self._country_code)
        patient_id = get_individual_id(self._hospital_name, department, patient)
        practitioner_id = get_individual_id(self._hospital_name, department, doctor)
        schedule_segments = convert_time_to_segment(self._START_HOUR, self._END_HOUR, self._TIME_UNIT, schedule)
        
        _id = get_appointment_id(practitioner_id, schedule_segments[0], schedule_segments[-1])
        start = get_iso_time(schedule[0], utc_offset=utc_offset)
        end = get_iso_time(schedule[-1], utc_offset=utc_offset)
        patient_ref = f'Patient/{patient_id}'
        practitioner_ref = f'Practitioner/{practitioner_id}'
        slot = [f'Slot/{practitioner_id}-slot{seg}' for seg in schedule_segments]

        return {
            'ID': _id,
            'START_HOUR': start,
            'END_HOUR': end,
            'SLOT': slot,
            'PATIENT': patient,
            'PATIENT_REF': patient_ref,
            'PRACTITIONER': doctor,
            'PRACTITIONER_REF': practitioner_ref,
        }


    def __call__(self, data_pair: Tuple[dict, dict], agent_test_data: dict, agent_results: dict, environment) -> dict:
        """
        Run the evaluation pipeline for generating and validating FHIR Appointment resources based on agent scheduling results.

        Args:
            data_pair (Tuple[dict, dict]): A pair of ground truth and patient data for agent simulation.
            agent_test_data (dict): Dictionary containing test metadata, doctor/patient/schedule data,
                                    and ground-truth information for each case.
            agent_results (dict): Dictionary containing scheduling results generated by the agent.
            environment (HospitalEnvironment): Hospital environment instance to manage patient schedules.

        Returns:
            dict: A dictionary with the following keys:
                - 'gt': List of ground-truth FHIR Appointment resources.
                - 'pred': List of predicted FHIR Appointment resources generated by the LLM.
                - 'status': List of boolean values indicating if each prediction passed the sanity check.
                - 'status_code': List of string codes representing the validation status.
        """
        gt, test_data = data_pair
        metadata = agent_test_data.get('metadata')
        self._hospital_name = metadata.get('hospital_name')
        self._country_code = metadata.get('country_code', 'KR')
        self._START_HOUR = metadata.get('time').get('start_hour')
        self._END_HOUR = metadata.get('time').get('end_hour')
        self._TIME_UNIT = metadata.get('time').get('interval_hour')
        schedule, sanity = self.__extract_schedules(gt, agent_results)
        results = self.get_result_dict()
        
        # Load ground truth Appointment FHIR resource
        gt_resource = self._get_fhir_appointment(self._fhir_data_path / 'appointment' / gt.get('fhir_resource'))
        results['gt'].append(gt_resource)

        # If the precedent schedule data is invalid, continue
        if not sanity:
            results['pred'].append({})
            results['status'].append(False)
            results['status_code'].append(self.status_codes['preceding'])
            return results

        # LLM call and compare the validity of the LLM output
        user_prompt = self.user_prompt_template.format(**self.__get_necessary_information(schedule))
        prediction = self.client(
            user_prompt,
            system_prompt=self.system_prompt, 
            using_multi_turn=False
        )
        prediction = MakeFHIRResource.postprocessing(prediction)
        expected_prediction = self._get_fhir_appointment(data={'metadata': deepcopy(metadata), 'information': deepcopy(schedule)})
        status, status_code, prediction = self._sanity_check(prediction, expected_prediction)
        
        # Append results
        results['pred'].append(prediction)
        results['status'].append(status)
        results['status_code'].append(status_code)
            
        return results



class MakeFHIRAPI(Task):
    def __init__(self, config):
        self.name = 'fhir_api'
        self.__init_env(config)
        self.system_prompt = txt_load(self._system_prompt_path)
        self.user_prompt_template = txt_load(self._user_prompt_path)
        self.client = GeminiClient(config.model) if 'gemini' in config.model.lower() else GPTClient(config.model)


    def __init_env(self, config):
        """
        Initialize necessary variables.

        Args:
            config (Config): Configuration for agent tasks.
        """
        self._system_prompt_path = config.fhir_api_task.system_prompt
        self._user_prompt_path = config.fhir_api_task.user_prompt
        self._fhir_data_path = Path(config.fhir_data)
        self._fhir_url = config.fhir_url


    @staticmethod
    def postprocessing(text: str) -> Union[str, None]:
        """
        Extracts a Bash command from a markdown-formatted code block in the given text.

        Args:
            text (str): The input text potentially containing a markdown-formatted Bash code block.

        Returns:
            Union[str, None]: The extracted Bash command if found, otherwise None.
        """
        try:
            match = re.search(r'```bash\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return match.group(1)
        except:
            return None
        

    def __extract_fhir_resource(self, gt: dict, agent_results: dict) -> Tuple[list[dict], list[bool]]:
        """
        Extracts the predicted FHIR Appointment resource from agent results.
        If predictions are not available, falls back to using ground truth labels.

        Args:
            data_pair (Tuple[dict, dict]): A pair of (ground_truth, test_data) for a single patient.
            gt (dict): Ground truth data of a patient.

        Returns:
            Tuple[list[dict], list[bool]]: A list of fhir_resources, either predicted or ground truth and each sanity status.
        """
        try:
            fhir_resource = agent_results['fhir_resource']['pred'][-1]
            sanity = agent_results['fhir_resource']['status'][-1]
        except:
            log('The predicted fhir_resource is not given. The ground truth value will be used.', 'warning')
            fhir_resource_path = self._fhir_data_path / 'appointment' / gt.get('fhir_resource')
            
            # Load the FHIR json file if it exists, otherwise create a new one
            fhir_resource = self._get_fhir_appointment(
                gt_resource_path=fhir_resource_path,
                data={'metadata': deepcopy(self._metadata), 'information': deepcopy(gt)}
            )
            sanity = True
        

        return fhir_resource, sanity
    

    def _sanity_check(self,
                      prediction: Union[str, None], 
                      expected_prediction: str) -> Tuple[bool, str, Union[str, None]]:
        """
        Perform a sanity check to compare the predicted curl command against the expected one.

        Args:
            prediction (Union[str, None]): The LLM-generated curl command.
            expected_prediction (str): The expected curl command.

        Returns:
            Tuple[bool, str, Union[str, None]]:
                - A boolean indicating whether the prediction passed the sanity check.
                - A status code string (e.g., 'correct', 'format') from `self.status_codes`.
                - The original prediction, for logging or further processing.
        """
        def check_curl(curl_command: str) -> bool:
            return curl_command.startswith('curl -X PUT')

        def extract_url(curl_command: str) -> str:
            match = re.search(r'(http[s]?://[^\s"]+)', curl_command)
            return match.group(0).strip() if match else ''

        def extract_headers(curl_command: str) -> set:
            pattern = r'-(?:H|-header) "([^"]+)"'
            return set(re.findall(pattern, curl_command))
        
        def extract_payload(curl_command: str) -> dict:
            pattern = r"-(?:d|-data)\s+(['\"])(\{.*?\})\1"  # ' 또는 "로 감싼 JSON 추출
            match = re.search(pattern, curl_command, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(2))
                except json.JSONDecodeError:
                    pass
            return {}
        
        # Check the prediciton format
        if not isinstance(prediction, str):
            return False, self.status_codes['format'], prediction    # Could not be parsed as a dictionary

        # Check the sanity of the generated command
        if not check_curl(prediction):
            return False, self.status_codes['curl']['http'], prediction
        if not extract_url(prediction) == extract_url(expected_prediction):
            return False, self.status_codes['curl']['url'], prediction
        if not extract_headers(prediction) == extract_headers(expected_prediction):
            return False, self.status_codes['curl']['header'], prediction
        if not self.sanity_check_recursively(extract_payload(prediction), extract_payload(expected_prediction)):
            return False, self.status_codes['curl']['payload'], prediction
        
        return True, self.status_codes['correct'], prediction
    

    def __get_api(self, resource: dict) -> str:
        """
        Generate curl API command from a FHIR resource.

        Args:
            resource (dict): FHIR resource ditionary.

        Returns:
            str: The generated curl API command.
        """
        gt_api = f'curl -X PUT "{self._fhir_url}/Appointment/{resource["id"]}" ' \
                 f'-H "Content-Type: application/fhir+json" ' \
                 f'-H "Accept: application/fhir+json" ' \
                 f"-d '{json.dumps(resource)}'"
        return gt_api


    def __call__(self, data_pair: Tuple[dict, dict], agent_test_data: dict, agent_results: dict, environment) -> dict:
        """
        Processes test data and LLM results to generate and validate FHIR Appointment resources.
        
        Args:
            data_pair (Tuple[dict, dict]): A pair of ground truth and patient data for agent simulation.
            agent_test_data (dict): Input test data containing metadata and agent-specific data for processing.
            agent_results (dict): Results produced by the agent (e.g., LLM output) to be validated.
            environment (HospitalEnvironment): Hospital environment instance to manage patient schedules.

        Returns:
            dict: A dictionary containing lists for each processed item with keys:
                - 'gt': ground truth FHIR resource API commands.
                - 'pred': predicted FHIR resource API commands from the LLM.
                - 'status': boolean flags indicating pass/fail of sanity checks.
                - 'status_code': human-readable status codes explaining the validation result.
        """
        gt, test_data = data_pair
        self._metadata = agent_test_data.get('metadata')
        fhir_resource, sanity = self.__extract_fhir_resource(gt, agent_results)
        results = self.get_result_dict()
        
        # Append a ground truth Appointment FHIR resource and API command
        gt_api = self.__get_api(
            self._get_fhir_appointment(
                gt_resource_path=self._fhir_data_path / 'appointment' / gt.get('fhir_resource'),
                data={'metadata': deepcopy(self._metadata), 'information': deepcopy(gt)}
            )
        )
        results['gt'].append(gt_api)

        # If the precedent resource data is invalid, continue
        if not sanity:
            results['pred'].append('')
            results['status'].append(False)
            results['status_code'].append(self.status_codes['preceding'])
            return results

        # LLM call and compare the validity of the LLM output
        user_prompt = self.user_prompt_template.format(FHIR_URL=self._fhir_url, RESOURCE=json.dumps(fhir_resource))
        prediction = self.client(
            user_prompt,
            system_prompt=self.system_prompt, 
            using_multi_turn=False
        )
        prediction = MakeFHIRAPI.postprocessing(prediction)
        expected_prediction = self.__get_api(fhir_resource)
        status, status_code, prediction = self._sanity_check(prediction, expected_prediction)
        
        # Append results
        results['pred'].append(prediction)
        results['status'].append(status)
        results['status_code'].append(status_code)
            
        return results
