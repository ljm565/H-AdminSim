import os
import json
import random
from copy import deepcopy
from decimal import getcontext
from importlib import resources
from typing import Tuple, Union, Optional
from dotenv import load_dotenv, find_dotenv

from patientsim import PatientAgent

from h_adminsim import SchedulingAdminStaffAgent
from h_adminsim.task import OutpatientTask
from h_adminsim.environment import OPFUSchedulingSimulation
from h_adminsim.environment.hospital import HospitalEnvironment
from h_adminsim.tools import DataConverter
from h_adminsim.tools.sanity_checker import SanityChecker
from h_adminsim.registry import (
    STATUS_CODES, 
    SCHEDULE_STATUS,
    OPFU_PREFERENCE_PHRASE_PATIENT,
)
from h_adminsim.utils import colorstr, log
from h_adminsim.utils.fhir_utils import *
from h_adminsim.utils.common_utils import *



class OutpatientFollowUpScheduling(OutpatientTask):
    def __init__(self, 
                 patient_model: str,
                 admin_staff_model: str,
                 preference_rejection_prob: float = 0.3,
                 preference_rejection_prob_decay: float = 0.5,
                 fhir_integration: bool = False,
                 scheduling_max_inference: int = 5,
                 scheduling_strategy: str = 'tool_calling',
                 max_retries: int = 8,
                 patient_vllm_endpoint: Optional[str] = None,
                 admin_staff_vllm_endpoint: Optional[str] = None):
        super().__init__()

        # Initialize variables
        getcontext().prec = 10
        dotenv_path = find_dotenv(usecwd=True)
        load_dotenv(dotenv_path, override=True)
        self.name = 'follow_up_visit_scheduling'
        self.patient_model, self.patient_vllm_endpoint, self.patient_use_vllm \
            = self._init_task_models(patient_model, patient_vllm_endpoint)
        self.admin_staff_model, self.admin_staff_vllm_endpoint, self.admin_staff_use_vllm \
            = self._init_task_models(admin_staff_model, admin_staff_vllm_endpoint)
        
        # Initialize scheduling methods and a staff agent
        self.admin_staff_agent = SchedulingAdminStaffAgent(
            target_task='follow_up_visit_scheduling',
            model=self.admin_staff_model,
            use_vllm=self.admin_staff_use_vllm,
            vllm_endpoint=self.admin_staff_vllm_endpoint,
            temperature=0 if not 'gpt-5' in self.admin_staff_model.lower() else 1
        )

        # Scheduling parameters
        self.preference_rejection_prob = preference_rejection_prob
        self.preference_rejection_prob_decay = preference_rejection_prob_decay

        # Others
        self.fhir_integration = fhir_integration
        self.max_retries = max_retries
        self.max_inferences = scheduling_max_inference
        self.scheduling_strategy = scheduling_strategy
        assert self.scheduling_strategy in ['reasoning', 'tool_calling'], \
            colorstr("red", 'Scheduling strategy must be either `reasoning` or `tool_calling`')
        self.schedule_patient_system_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath('opfu_schedule_patient_system.txt'))
        self.patient_reasoning_kwargs = {'reasoning_effort': 'low'} if 'gpt-5' in self.patient_model.lower() else {}
        self.staff_reasoning_kwargs = {'reasoning_effort': 'low'} if 'gpt-5' in self.admin_staff_model.lower() else {}

    
    def _init_simulation(self,
                         system_prompt_path: str,
                         environment: HospitalEnvironment,
                         additional_patient_conditions: dict = {}) -> OPFUSchedulingSimulation:
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
        sim_environment = OPFUSchedulingSimulation(
            patient_agent=patient_agent, 
            admin_staff_agent=self.admin_staff_agent, 
            metadata=self._metadata,
            department_data=self._department_data,
            environment=environment,
            scheduling_strategy=self.scheduling_strategy,
            preference_rejection_prob=self.preference_rejection_prob,
            preference_rejection_prob_decay=self.preference_rejection_prob_decay,
            fhir_integration=self.fhir_integration,
            sanity_checker=self.sanity_checker, 
        )
        return sim_environment


    def get_first_visit_patient_information(self, 
                                            gt: dict, 
                                            environment: HospitalEnvironment) -> Tuple[int, Optional[dict]]:
        """
        Extracts the patient name and predicted department from agent results.
        If predictions are not available, falls back to using ground truth labels.

        Args:
            gt (dict): Ground truth data of a patient.
            environment (HospitalEnvironment): Hospital environment.
            agent_test_data (dict): Dictionary containing test data and metadata for a single hospital.

        Returns:
            Tuple[int, Optional[dict]]: Patient information that assumed already saved in HIS and its index.
        """
        name = gt['patient']
        for idx, fv_schedule in enumerate(environment.patient_schedules):
            if fv_schedule['visit_type'] == 'first_visit' and fv_schedule['patient'] == name and fv_schedule['status'] == SCHEDULE_STATUS['completed']:
                return idx, fv_schedule
        return -1, None


    def cancellation_request(self, 
                        doctor_information: dict, 
                        environment: HospitalEnvironment, 
                        idx: Optional[int] = None, 
                        verbose: bool = False) -> Tuple[dict, Optional[dict]]:
        """
        Cancel a doctor's scheduled appointment.

        Args:
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                       including availability and other relevant details.
            environment (HospitalEnvironment): Hospital environment.
            idx (int, optional): Specific patient schedule index.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.

        Returns:
            Tuple[dict, Optional[dict]]: Updated doctor information and a result dictionary after cancellation.
        """
        if idx is None:
            candidate_idx = [i for i, schedule in enumerate(environment.patient_schedules) if schedule['status'] == SCHEDULE_STATUS['scheduled']]
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

            # Successfully cancelled
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

        return doctor_information, None
                

    def rescheduling_request(self,
                             doctor_information: dict,
                             environment: HospitalEnvironment, 
                             idx: Optional[int] = None, 
                             verbose: bool = False) -> Tuple[dict, Optional[dict]]:
        """
        Add a patient schedule to the waiting list in the given environment.

        Args:
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                       including availability and other relevant details.
            environment (HospitalEnvironment): Hospital environment.
            idx (int, optional): Specific patient schedule index.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.
        
        Returns:
            Tuple[dict, Optional[dict]]: Updated doctor information and a result dictionary after cancellation.
        """
        result_dict = init_result_dict()
        if idx is None:
            candidate_idx = [i for i, schedule in enumerate(environment.patient_schedules) if schedule['status'] == SCHEDULE_STATUS['scheduled']]
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
                doctor_information, result_dict = run_with_retry(
                    sim_environment.rescheduling_simulate,
                    gt_idx=idx,
                    doctor_information=doctor_information,
                    patient_schedules=environment.patient_schedules,
                    verbose=verbose,
                    max_inferences=self.max_inferences,
                    patient_kwargs=self.patient_reasoning_kwargs,
                    staff_kwargs=self.staff_reasoning_kwargs,
                    max_retries=self.max_retries,
                )

                if result_dict['status'][0] is not False:   # No GT and correct case
                    if 'patient' in result_dict['pred'][0]:
                        new_schedule = result_dict['pred'][0]
                        doctor_information[new_schedule['attending_physician']]['schedule'][new_schedule['date']].append(new_schedule['schedule'])
                        doctor_information[new_schedule['attending_physician']]['schedule'][new_schedule['date']].sort()
                        self.update_env(
                            status=True,
                            prediction=new_schedule,
                            environment=environment,
                        )

                return doctor_information, result_dict

            return doctor_information, None

        return doctor_information, None
            

    def _record_prescribed_tests_on_fv_appointment(self,
                                                   environment: HospitalEnvironment,
                                                   fv_patient_info: dict,
                                                   required_test_list: list,
                                                   code_to_test_name: dict):
        """
        Append HealthcareService references for the prescribed tests onto the completed
        first-visit consultation Appointment via `supportingInformation` (FHIR R5).

        Args:
            environment (HospitalEnvironment): Hospital environment.
            fv_patient_info (dict): First-visit patient schedule.
            required_test_list (list): Required test list after first-visit consultation.
            code_to_test_name (dict): Code to test name dictionary.
        """
        hospital_name = self._metadata['hospital_name']
        department = fv_patient_info['department']
        attending_physician = fv_patient_info['attending_physician']
        practitioner_id = get_individual_id(
            hospital_name,
            self._department_data[department]['code'].lower(),
            attending_physician,
        )

        schedule = fv_patient_info['schedule']
        if isinstance(schedule, dict) and 'time' in schedule:
            schedule = schedule['time']
        schedule_segments = convert_time_to_segment(
            self._START_HOUR, self._END_HOUR, self._TIME_UNIT, schedule,
        )
        appointment_id = get_appointment_id(
            practitioner_id,
            fv_patient_info['date'],
            schedule_segments[0],
            schedule_segments[-1],
        )

        appointment = environment.fhir_manager.read('Appointment', appointment_id, verbose=False).json()
        existing = appointment.get('supportingInformation', [])
        existing_refs = {entry.get('reference') for entry in existing if entry.get('reference')}
        for test in required_test_list:
            ref = f"HealthcareService/{get_healthcareservice_id(hospital_name, test['test_code'])}"
            if ref in existing_refs:
                continue
            existing.append({
                'reference': ref,
                'display': code_to_test_name.get(test['test_code'], test['test_code']),
            })
            existing_refs.add(ref)
        appointment['supportingInformation'] = existing
        environment.fhir_manager.update('Appointment', appointment_id, appointment, verbose=False)


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
            

    def __call__(self, 
                 data_pair: Tuple[dict, dict], 
                 agent_test_data: dict, 
                 agent_results: dict, 
                 environment: HospitalEnvironment, 
                 verbose: bool = False,
                 **kwargs) -> dict:
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
        self.sanity_checker = SanityChecker(self._START_HOUR, self._END_HOUR, self._TIME_UNIT)
        doctor_information = environment.get_general_doctor_info_from_fhir() if self.fhir_integration else agent_test_data.get('doctor')
        test_information = environment.get_general_test_info_from_fhir() if self.fhir_integration else agent_test_data.get('test')
        gt_idx, fv_patient_info = self.get_first_visit_patient_information(gt, environment)
        code_to_test_name = {test['code']: test['name'] for _, tests in test_information.items() for test in tests}
        results = init_result_dict()
        self.reset_token_data()

        # First-visit identity from the actually-scheduled first-visit entry
        assert fv_patient_info is not None, \
            colorstr("red", f"First-visit entry for {gt['patient']} not found in patient_schedules")
        assert fv_patient_info['attending_physician'] in self._department_data[fv_patient_info['department']]['doctor'], \
            colorstr("red", "Attending physician must belong to the primary department in follow up case")
        attending_physician = fv_patient_info['attending_physician']
        department = fv_patient_info['department']
        required_test_list = gt['required_tests']

        # Assuming the test was determined by the doctor
        fv_patient_info['test'] = required_test_list
        if self.fhir_integration:
            self._record_prescribed_tests_on_fv_appointment(
                environment=environment,
                fv_patient_info=fv_patient_info,
                required_test_list=required_test_list,
                code_to_test_name=code_to_test_name,
            )

        # Make scheduling GT list
        gt_data = [
            {   
                'index': gt_idx,
                'patient_fv': fv_patient_info,
                'department': department,
                'attending_physician': attending_physician,
                'required_tests': required_test_list,
                'preference': preference,
            } for preference in gt.get('preference')
        ]

        #################################################### Regular Scheudling Simulation ####################################################
        # Initialize the simulation environment using the first preference data
        staff_known_data = {
            'patient_fv': None,
            'department': None,
            'attending_physician': None,
            'required_tests': None,
            'patient_intention': None,
        }
        preference = gt_data[0].get('preference')
        preference_desc = OPFU_PREFERENCE_PHRASE_PATIENT[preference]
        required_test_desc = [f"{i+1}. {code_to_test_name[test['test_code']]}" for i, test in enumerate(gt_data[0]['required_tests'])]
        sim_environment = self._init_simulation(
            system_prompt_path=self.schedule_patient_system_prompt_path,
            environment=environment,
            additional_patient_conditions={
                'doctor': gt_data[0]['attending_physician'],
                'tests': required_test_desc,
                'preference': preference,
                'preference_desc': preference_desc,
                'name': gt['patient'],
                'gender': gt['gender'],
                'birth_date': gt['birthDate'],
                'telecom': gt['telecom'][0]['value'],
                'personal_id': gt['identifier'][0]['value'],
                'address': gt['address'][0]['text'],
            }
        )
    
        # Simulate the main scheduling task
        doctor_information, test_information, result_dict, token_usage = run_with_retry(
            sim_environment.test_scheduling_simulate,
            gt_data=gt_data,
            staff_known_data=staff_known_data,
            doctor_information=doctor_information,
            test_device_information=test_information,
            verbose=verbose,
            patient_kwargs=self.patient_reasoning_kwargs,
            staff_kwargs=self.staff_reasoning_kwargs,
            max_retries=self.max_retries,
        )
        self.save_token_data(
            token_usage['patient_token'], 
            token_usage['admin_staff_token'], 
        )

        prediction, status, status_code = \
            result_dict['pred'][0], result_dict['status'][0], result_dict['status_code'][0]
        
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
        for key in result_dict.keys():
            results[key] += result_dict[key]
        results['token'].append(self.token_stats)
        #######################################################################################################################################
        
        # Other events
        ## Simulate the schedule cancellation requests
        if random.random() < self.schedule_cancellation_prob:
            doctor_information, result_dict = self.cancellation_request(
                doctor_information=doctor_information,
                environment=environment,
                verbose=verbose,
            )
            if result_dict is not None:
                agent_test_data['doctor'] = doctor_information
                results['gt'].extend(result_dict['gt'])
                results['pred'].extend(result_dict['pred'])
                results['status'].extend(result_dict['status'])
                results['status_code'].extend(result_dict['status_code'])
                results['dialog'].extend(result_dict['dialog'])
                results['token'].extend([{}]*len(result_dict['gt']))

                if verbose:
                    log(f'Pred  : {result_dict["pred"]}')
                    log(f'Status: {result_dict["status_code"]}')
                    log(f'Final Status: {result_dict["status_code"]}\n\n\n')
        
        ## Simulate the resecheduling requests
        if random.random() < self.request_early_schedule_prob:
            doctor_information, result_dict = self.rescheduling_request(
                doctor_information=doctor_information,
                environment=environment, 
                verbose=verbose
            )
            if result_dict is not None:
                agent_test_data['doctor'] = doctor_information
                results['gt'].extend(result_dict['gt'])
                results['pred'].extend(result_dict['pred'])
                results['status'].extend(result_dict['status'])
                results['status_code'].extend(result_dict['status_code'])
                results['dialog'].extend(result_dict['dialog'])
                results['token'].extend([{}]*len(result_dict['gt']))

                if verbose:
                    log(f'Pred  : {result_dict["pred"]}')
                    log(f'Status: {result_dict["status_code"]}')
                    log(f'Final Status: {result_dict["status_code"]}\n\n\n')

        return results
