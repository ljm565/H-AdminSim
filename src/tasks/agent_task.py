import json
from pathlib import Path
from copy import deepcopy
from typing import Tuple, Union

from tools import GeminiClient, GPTClient, DataConverter
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
        'preceding': 'preceding task failed',
        'correct': 'pass',
    }

    def get_result_dict(self):
        return {'gt': [], 'pred': [], 'status': [], 'status_code': []}



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


    def __call__(self, agent_test_data: dict, agent_results: dict) -> dict:
        """
        Estimates the most appropriate medical department for each patient using an LLM agent.

        Args:
            agent_test_data (dict): A dictionary containing test data for a single hospital.
                Expected to include:
                    - 'agent_data': List of (ground_truth, test_data) pairs.
                    - 'department': Dictionary of available departments.
            agent_results (dict): Placeholder for compatibility; not used in this method.

        Returns:
            dict: A dictionary with:
                - 'gt': List of ground-truth departments.
                - 'pred': List of predicted departments from the LLM agent.
                - 'status': List of booleans indicating whether each prediction correct.
                - 'status_code': List of status codes explaining each status.
        """
        agent_data = agent_test_data['agent_data']
        departments = list(agent_test_data['department'].keys())
        options = ''.join([f'{i+1}. {department}\n' for i, department in enumerate(departments)])
        results = self.get_result_dict()
        
        for data_pair in agent_data:
            gt, test_data = data_pair
            gt_department = gt['department']
            results['gt'].append(gt_department)
            
            # LLM call
            user_prompt = self.user_prompt_template.format(SYMPTOM=test_data['symptom'], OPTIONS=options)
            output = self.client(
                user_prompt,
                system_prompt=self.system_prompt, 
                using_multi_turn=False
            )
            output = AssignDepartment.postprocessing(output)
            
            # Append results
            status = gt_department == output
            status_code = self.status_codes['correct'] if status else self.status_codes['department']
            results['pred'].append(output)
            results['status'].append(status)
            results['status_code'].append(status_code)
        
        return results



class AssignSchedule(Task):
    def __init__(self, config):
        self.name = 'schedule'
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
        self._system_prompt_path = config.schedule_task.system_prompt
        self._user_prompt_path = config.schedule_task.user_prompt

    
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
            text = json.loads(text)
            key = list(text.keys())[0]
            text[key]['start'] = float(text[key]['start'])
            text[key]['end'] = float(text[key]['end'])
            return text
        except:
            return text


    def __extract_departments(self, agent_data: list[Tuple[dict, dict]], agent_results: dict) -> Tuple[list[str], list[bool]]:
        """
        Extracts the predicted department from agent results.
        If predictions are not available, falls back to using ground truth labels.

        Args:
            agent_data (list[Tuple[dict, dict]]): A list of (ground_truth, test_data) pairs for each patient.
            agent_results (dict): A dictionary that may contain predicted department results under the key 'department'.

        Returns:
            Tuple[list[str], list[bool]]: A list of departments, either predicted or ground truth and each sanity status.
        """
        try:
            departments = agent_results['department']['pred']
            sanities = agent_results['department']['status']
        except:
            log('Predicted departments are not given. Ground truth values will be used.', 'warning')
            departments = [gt['department'] for gt, _ in agent_data]
            sanities = [True] * len(departments)
        
        assert len(departments) == len(sanities) == len(agent_data), log('The number of departments does not match the agent data.', 'error')

        return departments, sanities
    

    def _sanity_check(self,
                      prediction: Union[str, dict], 
                      patient_condition: dict,
                      doctor_information: dict) -> Tuple[bool, str, Union[str, dict]]:
        """
        Validates a predicted schedule for a doctor by checking its structure, time validity, 
        duplication with existing schedules, and updates the doctor's schedule if valid.

        Args:
            prediction (Union[str, dict]): The predicted allocation result, either a string (if parsing failed)
                or a dictionary mapping a doctor's name to a schedule with 'start' and 'end' times.
            patient_condition (dict): The conditions including duration, etc.
            doctor_information (dict): Dictionary of doctor data including their existing schedules.
                Each key is a doctor's name, and each value includes a 'schedule' field.

        Returns:
            Tuple[bool, str, Union[str, dict]]: 
                - A boolean indicating whether the prediction passed all sanity checks.
                - A string explaining its status.
                - The original prediction (either unchanged or used for debugging/logging if invalid).
        """
        # Check the prediciton format
        if not isinstance(prediction, dict):
            return False, self.status_codes['format'], prediction    # Could not be parsed as a dictionary
        elif len(prediction) > 1:
            return False, self.status_codes['conflict']['physician'], prediction    # Allocated more than one doctor; cannot determine target
        
        # Check the predicted schedule type and validities
        try:
            doctor_name = list(prediction.keys())[0]
            start = prediction[doctor_name]['start']
            end = prediction[doctor_name]['end']
            fixed_schedules = doctor_information[doctor_name]['schedule']
            assert isinstance(start, float) and isinstance(end, float) \
                and start < end and start >= self._START_HOUR and end <= self._END_HOUR
            assert patient_condition['department'] == doctor_information[doctor_name]['department'] \
                and patient_condition['duration'] == round(end - start, 4)
        except KeyError:
            return False, self.status_codes['format'], prediction    # Schedule allocation missing or doctor not found
        except AssertionError:
            return False, self.status_codes['schedule'], prediction    # Invalid schedule times or department

        # Check the duplication of the schedules
        prediction_schedule_segemnts = convert_time_to_segment(self._START_HOUR,
                                                               self._END_HOUR,
                                                               self._TIME_UNIT,
                                                               [start, end])
        fixed_schedule_segments = sum([convert_time_to_segment(self._START_HOUR, 
                                                          self._END_HOUR, 
                                                          self._TIME_UNIT, 
                                                          fs) for fs in fixed_schedules], [])
        
        if len(set(prediction_schedule_segemnts) & set(fixed_schedule_segments)):
            return False, self.status_codes['conflict']['time'], prediction    # Overlaps with an existing schedule
        
        # Finally update schedule of the doctor
        doctor_information[doctor_name]['schedule'].append([start, end])    # In-place logic
        prediction = {
            'patient': patient_condition.get('patient'),
            'attending_physician': doctor_name,
            'department': patient_condition.get('department'),
            'schedule': [start, end]
        }
        return True, self.status_codes['correct'], prediction
            

    def __call__(self, agent_test_data: dict, agent_results: dict) -> dict:
        """
        This method uses agent test data to prompt an LLM for scheduling decisions, post-processes
        the output, runs sanity checks on predicted schedules, and collects the results for evaluation.

        Args:
            agent_test_data (dict): Dictionary containing test data and metadata for a single hospital.
                Expected keys include:
                    - 'metadata': A dict containing start_hour, end_hour, and interval_hour under 'time'.
                    - 'agent_data': A list of (ground_truth, test_data) pairs.
                    - 'doctor': A dictionary of doctor profiles with department and schedule info.
            agent_results (dict): Optional dictionary containing prior department predictions.
                Used to extract department-level guidance per patient. Can be empty.

        Returns:
            dict: A dictionary with three keys:
                - 'gt': List of ground truth results, each including patient info, attending physician, department, and schedule.
                - 'pred': List of predicted results (either valid dict or fallback string).
                - 'status': List of booleans indicating whether each prediction passed sanity checks.
                - 'status_code': List of status codes explaining each status.
        """
        self._START_HOUR = agent_test_data.get('metadata').get('time').get('start_hour')
        self._END_HOUR = agent_test_data.get('metadata').get('time').get('end_hour')
        self._TIME_UNIT = agent_test_data.get('metadata').get('time').get('interval_hour')
        agent_data = agent_test_data.get('agent_data')
        doctor_information = agent_test_data.get('doctor')
        departments, sanities = self.__extract_departments(agent_data, agent_results)
        results = self.get_result_dict()
        
        for data_pair, department, sanity in zip(agent_data, departments, sanities):
            gt, test_data = data_pair
            gt_results = {
                'patient': gt.get('patient'),
                'attending_physician': gt.get('attending_physician'),
                'department': gt.get('department'),
                'schedule': gt.get('schedule').get('time')
            }
            results['gt'].append(gt_results)

            # If the department data is wrong, continue
            if not sanity:
                results['pred'].append({})
                results['status'].append(False)
                results['status_code'].append(self.status_codes['preceding'])
                continue
            
            # LLM call
            duration = test_data.get('constraint').get('duration')
            doctor_information_str = json.dumps(doctor_information, indent=2)   # String-converted ditionary 
            user_prompt = self.user_prompt_template.format(
                START_HOUR=self._START_HOUR,
                END_HOUR=self._END_HOUR,
                TIME_UNIT=self._TIME_UNIT,
                DEPARTMENT=department,
                DURATION=duration,
                DOCTOR=doctor_information_str
            )
            output = self.client(
                user_prompt,
                system_prompt=self.system_prompt, 
                using_multi_turn=False
            )
            output = AssignSchedule.postprocessing(output)
            status, status_code, output = self._sanity_check(
                output, 
                {'patient': test_data.get('patient'), 'department': department, 'duration': duration},
                doctor_information
            )
            
            # Append results
            results['pred'].append(output)
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
            
    
    def __extract_schedules(self, agent_data: list[Tuple[dict, dict]], agent_results: dict) -> Tuple[list[dict], list[bool]]:
        """
        Extracts the predicted schedule from agent results.
        If predictions are not available, falls back to using ground truth labels.

        Args:
            agent_data (list[Tuple[dict, dict]]): A list of (ground_truth, test_data) pairs for each patient.
            agent_results (dict): A dictionary that may contain predicted schedule results under the key 'schedule'.

        Returns:
            Tuple[list[str], list[bool]]:: A list of scedules, either predicted or ground truth and each sanity status.
        """
        try:
            schedules = agent_results['schedule']['pred']
            sanities = agent_results['schedule']['status']
        except:
            log('Predicted schedules are not given. Ground truth values will be used.', 'warning')
            schedules = [{'patient': gt['patient'], 
                          'attending_physician': gt['attending_physician'],
                          'department': gt['department'],
                          'schedule': gt['schedule']['time']} for gt, _ in agent_data]
            sanities = [True] * len(schedules)

        assert len(schedules) == len(sanities) == len(agent_data), log('The number of schedules does not match the agent data.', 'error')

        return schedules, sanities
    

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
        def sanity_check_recursively(prediction, expected_prediction):
            if isinstance(expected_prediction, dict):
                if not isinstance(prediction, dict):
                    return False
                for key, val in expected_prediction.items():
                    if key not in prediction:
                        return False
                    if not sanity_check_recursively(prediction[key], val):
                        return False
                return True
            
            elif isinstance(expected_prediction, list):
                if not isinstance(prediction, list) or len(expected_prediction) != len(prediction):
                    return False
                
                unmatched = prediction.copy()
                for expected_item in expected_prediction:
                    matched = False
                    for i, pred_item in enumerate(unmatched):
                        if sanity_check_recursively(pred_item, expected_item):
                            unmatched.pop(i)
                            matched = True
                            break
                    if not matched:
                        return False
                return True

            else:
                return expected_prediction == prediction

        # Check the prediciton format
        if not isinstance(prediction, dict):
            return False, self.status_codes['format'], prediction    # Could not be parsed as a dictionary
        
        # Compare recursively
        if sanity_check_recursively(expected_prediction, prediction):
            return True, self.status_codes['correct'], prediction
        else:
            return False, self.status_codes['format'], prediction
    
        
    def __get_gt_resource(self, gt: dict) -> dict:
        """
        Load the ground-truth FHIR resource associated with the given GT (ground-truth) data.

        Args:
            gt (dict): A dictionary containing ground-truth scheduling information,
                       including the filename of the FHIR resource under the key 'fhir_resource'.

        Returns:
            dict: The loaded FHIR resource as a dictionary if the file exists.
                  Returns an empty dictionary if the file is not found.
        """
        gt_resource_path = self._fhir_data_path / 'appointment' / gt.get('fhir_resource')
        try:
            return json_load(gt_resource_path)
        except FileNotFoundError:
            return {}


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


    def __call__(self, agent_test_data: dict, agent_results: dict) -> dict:
        """
        Run the evaluation pipeline for generating and validating FHIR Appointment resources based on agent scheduling results.

        Args:
            agent_test_data (dict): Dictionary containing test metadata, doctor/patient/schedule data,
                                    and ground-truth information for each case.
            agent_results (dict): Dictionary containing scheduling results generated by the agent.

        Returns:
            dict: A dictionary with the following keys:
                - 'gt': List of ground-truth FHIR Appointment resources.
                - 'pred': List of predicted FHIR Appointment resources generated by the LLM.
                - 'status': List of boolean values indicating if each prediction passed the sanity check.
                - 'status_code': List of string codes representing the validation status.
        """
        self._hospital_name = agent_test_data.get('metadata').get('hospital_name')
        self._country_code = agent_test_data.get('metadata').get('country_code', 'KR')
        self._START_HOUR = agent_test_data.get('metadata').get('time').get('start_hour')
        self._END_HOUR = agent_test_data.get('metadata').get('time').get('end_hour')
        self._TIME_UNIT = agent_test_data.get('metadata').get('time').get('interval_hour')

        agent_data = agent_test_data.get('agent_data')
        schedules, sanities = self.__extract_schedules(agent_data, agent_results)
        results = self.get_result_dict()
        
        for data_pair, schedule, sanity in zip(agent_data, schedules, sanities):
            gt, test_data = data_pair
            
            # Load grount truth Appointment FHIR resource
            gt_resource = self.__get_gt_resource(gt)
            results['gt'].append(gt_resource)

            # If the schedule data is invalid, continue
            if not sanity:
                results['pred'].append({})
                results['status'].append(False)
                results['status_code'].append(self.status_codes['preceding'])
                continue

            # LLM call
            user_prompt = self.user_prompt_template.format(**self.__get_necessary_information(schedule))
            output = self.client(
                user_prompt,
                system_prompt=self.system_prompt, 
                using_multi_turn=False
            )
            output = MakeFHIRResource.postprocessing(output)
            expected_prediction = DataConverter.data_to_appointment(
                {
                    'metadata': deepcopy(agent_test_data.get('metadata')),
                    'patient': {
                        schedule.get('patient'): {
                            'department': schedule.get('department'),
                            'attending_physician': schedule.get('attending_physician'),
                            'schedule': schedule.get('schedule')
                        }
                    }
                }
            )[0]
            status, status_code, output = self._sanity_check(output, expected_prediction)
            
            # Append results
            results['pred'].append(output)
            results['status'].append(status)
            results['status_code'].append(status_code)
            
        return results



class MakeFHIRAPI(Task):
    def __init__(self, config):
        self.client = GeminiClient(config.model) if 'gemini' in config.model.lower() else GPTClient(config.model)


    def __call__(self):
        pass
