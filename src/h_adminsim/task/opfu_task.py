import os
import json
import random
from decimal import getcontext
from importlib import resources
from dotenv import load_dotenv, find_dotenv
from typing import Tuple, Union, Optional, TYPE_CHECKING

from patientsim import PatientAgent

from h_adminsim.task import OutpatientTask
from h_adminsim.simulation import OPFUSchedulingSimulation
from h_adminsim.tools.sanity_checker import SanityChecker
from h_adminsim.registry import (
    STATUS_CODES, 
    SCHEDULE_STATUS,
    OPFU_PREFERENCE_PHRASE_PATIENT,
)
from h_adminsim.registry.d_class import (
    StaffNegotiationPolicy,
    PatientNegotiationPolicy,
)
from h_adminsim.utils import colorstr, log
from h_adminsim.utils.mas_utils import *
from h_adminsim.utils.fhir_utils import *
from h_adminsim.utils.common_utils import *

if TYPE_CHECKING:
    from h_adminsim.pipeline import HospitalMAS
    from h_adminsim.environment.hospital import HospitalEnvironment



class OutpatientFollowUpScheduling(OutpatientTask):
    def __init__(self, 
                 patient_model: str,
                 admin_staff_mas: "HospitalMAS",
                 schedule_cancellation_prob: float = 0.05,
                 request_early_schedule_prob: float = 0.1,
                 preference_rejection_prob: float = 0.3,
                 preference_rejection_prob_decay: float = 0.5,
                 fhir_integration: bool = False,
                 scheduling_max_inference: int = 5,
                 scheduling_strategy: str = 'tool_calling',
                 max_retries: int = 8,
                 patient_vllm_endpoint: Optional[str] = None,
                 negotiation_params: dict = {}):
        super().__init__()

        # Initialize variables
        getcontext().prec = 10
        dotenv_path = find_dotenv(usecwd=True)
        load_dotenv(dotenv_path, override=True)
        self.name = 'follow_up_visit_scheduling'
        self.patient_model, self.patient_vllm_endpoint, self.patient_use_vllm \
            = init_task_models(patient_model, patient_vllm_endpoint)
        self.admin_staff_mas = admin_staff_mas
        
        # Scheduling parameters
        self.schedule_cancellation_prob = schedule_cancellation_prob
        self.request_early_schedule_prob = request_early_schedule_prob
        self.preference_rejection_prob = preference_rejection_prob
        self.preference_rejection_prob_decay = preference_rejection_prob_decay

        # Initialize the negotiation metrics for each preference
        self.staff_policy = StaffNegotiationPolicy(
            **negotiation_params
        )

        # Others
        self.fhir_integration = fhir_integration
        self.max_retries = max_retries
        self.max_inferences = scheduling_max_inference
        self.scheduling_strategy = scheduling_strategy
        assert self.scheduling_strategy in ['reasoning', 'tool_calling'], \
            colorstr("red", 'Scheduling strategy must be either `reasoning` or `tool_calling`')
        self.schedule_patient_system_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath('opfu_schedule_patient_system.txt'))
        self.cancel_patient_system_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath('opfu_cancel_patient_system.txt'))
        self.reschedule_patient_system_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath('opfu_reschedule_patient_system.txt'))
        self.patient_reasoning_kwargs = {'reasoning_effort': 'low'} if 'gpt-5' in self.patient_model.lower() else {}
        self.staff_reasoning_kwargs = {'reasoning_effort': 'low'} if 'gpt-5' in self.admin_staff_mas.get_agent(self.name).model.lower() else {}

    
    def _init_simulation(self,
                         system_prompt_path: str,
                         environment: "HospitalEnvironment",
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
            admin_staff_mas=self.admin_staff_mas, 
            metadata=self._metadata,
            department_data=self._department_data,
            environment=environment,
            scheduling_strategy=self.scheduling_strategy,
            preference_rejection_prob=self.preference_rejection_prob,
            preference_rejection_prob_decay=self.preference_rejection_prob_decay,
            fhir_integration=self.fhir_integration,
            sanity_checker=self.sanity_checker,
            negotiation_policy=self.staff_policy,
        )
        return sim_environment


    def get_first_visit_patient_information(self, 
                                            gt: dict, 
                                            environment: "HospitalEnvironment") -> Tuple[int, Optional[dict]]:
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
    

    def _book_schedule(self,
                       status: bool,
                       prediction: dict,
                       doctor_information: dict,
                       test_information: dict,
                       environment: "HospitalEnvironment"):
        """
        Book a follow-up prediction: append each test's device slot + the follow-up consultation slot, then sync the env.

        Args:
            status (bool): Whether the scheduling task was successful. If True, FHIR resources may be updated.
            prediction (dict): The follow-up booking to apply.
            doctor_information (dict): Doctor schedules to update in place.
            test_information (dict): Test device schedules to update in place.
            environment (HospitalEnvironment): Hospital environment.
        """
        if status:
            device_to_schedule = {device: info['schedule'] for _, tests in test_information.items() for test in tests for device, info in test['devices'].items()}
            
            # Test schedule
            for entry in prediction.get('test') or []:
                device, date, slot = entry['device'], entry['date'], entry['schedule']
                device_to_schedule[device][date].append(slot)
                device_to_schedule[device][date].sort()
            
            # Follow-up visit consultation
            if prediction['schedule']:
                doctor_information[prediction['attending_physician']]['schedule'][prediction['date']].append(prediction['schedule'])
                doctor_information[prediction['attending_physician']]['schedule'][prediction['date']].sort()
        
        self.update_env(
            status=status, 
            prediction=prediction, 
            environment=environment
        )


    def cancellation_request(self,
                             doctor_information: dict,
                             test_information: dict,
                             environment: "HospitalEnvironment",
                             idx: Optional[int] = None,
                             verbose: bool = False) -> Tuple[dict, dict, Optional[dict]]:
        """
        Cancel all of a patient's scheduled tests.

        Args:
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                       including availability and other relevant details.
            test_information (dict): A dictionary containing test device schedules.
            environment (HospitalEnvironment): Hospital environment.
            idx (int, optional): Specific patient schedule index.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.

        Returns:
            Tuple[dict, dict, Optional[dict]]: Updated doctor information, test information, and a result dictionary after cancellation.
        """
        # Candidates are follow-up bookings that carry prescribed tests ('not_yet' covers bookings whose follow-up slot was out of range)
        if idx is None:
            candidate_idx = [i for i, schedule in enumerate(environment.patient_schedules)
                             if schedule['visit_type'] == 'follow_up_visit'
                             and schedule['status'] in (SCHEDULE_STATUS['scheduled'], SCHEDULE_STATUS['not_yet'])
                             and schedule.get('test')]
            idx = random.choice(candidate_idx) if len(candidate_idx) else -1

        if idx >= 0:
            # Ground-truth cancelled booking
            cancelled_schedule = environment.patient_schedules[idx]
            patient = cancelled_schedule['patient']
            doctor = cancelled_schedule['attending_physician']

            # Initialize simulation environment for cancellation
            sim_environment = self._init_simulation(
                system_prompt_path=self.cancel_patient_system_prompt_path,
                environment=environment,
                additional_patient_conditions={
                    'patient_name': patient,
                    'doctor_name': doctor,
                }
            )

            # Test cancellation simulation
            doctor_information, test_information, result_dict = run_with_retry(
                sim_environment.test_canceling_simulate,
                gt_idx=idx,
                doctor_information=doctor_information,
                test_device_information=test_information,
                patient_schedules=environment.patient_schedules,
                verbose=verbose,
                max_inferences=self.max_inferences,
                patient_kwargs=self.patient_reasoning_kwargs,
                staff_kwargs=self.staff_reasoning_kwargs,
                max_retries=self.max_retries,
            )

            # Successfully cancelled
            if result_dict['status'][0] is not False:   # No GT and correct case
                # Update waiting list due to the freed slots
                doctor_information, test_information, rs_result_dict = self.automatic_waiting_list_update(
                    sim_environment=sim_environment,
                    environment=environment,
                    doctor_information=doctor_information,
                    test_information=test_information,
                )

                # Update result dictionary
                for key in result_dict.keys():
                    if len(rs_result_dict[key]):
                        result_dict[key].append(tuple(rs_result_dict[key]))

            return doctor_information, test_information, result_dict

        return doctor_information, test_information, None
                

    def rescheduling_request(self,
                             doctor_information: dict,
                             test_information: dict,
                             environment: "HospitalEnvironment",
                             idx: Optional[int] = None,
                             verbose: bool = False) -> Tuple[dict, dict, Optional[dict]]:
        """
        Move all of a patient's scheduled tests (and the follow-up consultation) to an earlier time.

        Args:
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                       including availability and other relevant details.
            test_information (dict): A dictionary containing test device schedules.
            environment (HospitalEnvironment): Hospital environment.
            idx (int, optional): Specific patient schedule index.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.

        Returns:
            Tuple[dict, dict, Optional[dict]]: Updated doctor information, test information, and a result dictionary after rescheduling.
        """
        result_dict = init_result_dict()
        # Candidates: follow-up bookings whose every test is still scheduled (none performed), not already waiting
        # ('not_yet' covers bookings whose follow-up slot was out of range; the per-test check still excludes performed tests)
        if idx is None:
            candidate_idx = [i for i, schedule in enumerate(environment.patient_schedules)
                             if schedule['visit_type'] == 'follow_up_visit'
                             and schedule['status'] in (SCHEDULE_STATUS['scheduled'], SCHEDULE_STATUS['not_yet'])
                             and schedule.get('test')
                             and all(t.get('status') == SCHEDULE_STATUS['scheduled'] for t in schedule['test'])
                             and all(schedule != s[1] for s in environment.waiting_list)]
            idx = random.choice(candidate_idx) if len(candidate_idx) else -1

        if idx >= 0:
            # Ground-truth rescheduling requested booking
            requested_schedule = environment.patient_schedules[idx]
            patient = requested_schedule['patient']
            doctor = requested_schedule['attending_physician']

            # Initialize simulation environment for rescheduling request
            sim_environment = self._init_simulation(
                system_prompt_path=self.reschedule_patient_system_prompt_path,
                environment=environment,
                additional_patient_conditions={
                    'patient_name': patient,
                    'doctor_name': doctor,
                }
            )

            # Rescheduling request simulation
            doctor_information, test_information, result_dict = run_with_retry(
                sim_environment.test_rescheduling_simulate,
                gt_idx=idx,
                doctor_information=doctor_information,
                test_device_information=test_information,
                patient_schedules=environment.patient_schedules,
                verbose=verbose,
                max_inferences=self.max_inferences,
                patient_kwargs=self.patient_reasoning_kwargs,
                staff_kwargs=self.staff_reasoning_kwargs,
                max_retries=self.max_retries,
            )

            # Successfully rescheduled -> book the new (earlier) schedule
            if result_dict['status'][0] is not False:   # No GT and correct case
                if 'patient' in result_dict['pred'][0]:
                    self._book_schedule(True, result_dict['pred'][0], doctor_information, test_information, environment)

            return doctor_information, test_information, result_dict

        return doctor_information, test_information, None
    

    def automatic_waiting_list_update(self,
                                      sim_environment: OPFUSchedulingSimulation,
                                      environment: "HospitalEnvironment",
                                      doctor_information: dict,
                                      test_information: dict) -> Tuple[dict, dict, dict]:
        """
        Automatically update the waiting list by attempting to move waiting test sets earlier.

        Args:
            sim_environment (OPFUSchedulingSimulation): The simulation environment used for scheduling.
            environment (HospitalEnvironment): Hospital environment.
            doctor_information (dict): A dictionary containing information about the doctor(s).
            test_information (dict): A dictionary containing test device schedules.

        Returns:
            Tuple[dict, dict, dict]: Updated doctor information, test information, and a result dictionary.
        """
        all_result_dict = init_result_dict()
        for result in sim_environment.automatic_waiting_list_update(
            doctor_information=doctor_information,
            test_device_information=test_information,
            **self.staff_reasoning_kwargs,
        ):
            doctor_information, test_information, result_dict = result['doctor_information'], result['test_device_information'], result['result_dict']

            if result_dict['status'][0]:
                new_schedule, original = result_dict['pred'][0], result['original']
                self._book_schedule(True, new_schedule, doctor_information, test_information, environment)
                log(f'{colorstr("[RESCHEDULED]")}: {original} is rescheduled to {new_schedule}')

            all_result_dict['gt'].extend(result_dict['gt'])
            all_result_dict['pred'].extend(result_dict['pred'])
            all_result_dict['status'].extend(result_dict['status'])
            all_result_dict['status_code'].extend(result_dict['status_code'])
            all_result_dict['dialog'].extend(result_dict['dialog'])

        return doctor_information, test_information, all_result_dict


    def _record_prescribed_tests_on_fv_appointment(self,
                                                   environment: "HospitalEnvironment",
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
            ref = f"HealthcareService/{get_healthcareservice_id(hospital_name, test['code'])}"
            if ref in existing_refs:
                continue
            existing.append({
                'reference': ref,
                'display': code_to_test_name.get(test['code'], test['code']),
            })
            existing_refs.add(ref)
        appointment['supportingInformation'] = existing
        environment.fhir_manager.update('Appointment', appointment_id, appointment, verbose=False)


    def update_env(self,
                   status: bool,
                   prediction: Union[dict, str],
                   environment: "HospitalEnvironment",
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
        fhir_test_appointments = []
        if status and self.fhir_integration:
            if patient_information is not None:
                fhir_patient = self.get_patient_fhir_resource(
                    self._metadata,
                    self._department_data,
                    patient_information,
                    prediction['department']
                )

            # To avoid None case of follow-up appointment
            if prediction['schedule']:
                fhir_appointment = self.get_appointment_fhir_resource(
                    self._metadata,
                    self._department_data,
                    prediction,
                )

            # Tests can be booked even when the follow-up consultation slot is out of range
            if prediction.get('test'):
                fhir_test_appointments = self.get_test_appointment_fhir_resource(
                    self._metadata,
                    self._department_data,
                    prediction,
                )

        appointments = ([fhir_appointment] if fhir_appointment else []) + fhir_test_appointments
        environment.update_env(
            status=status,
            patient_schedule=prediction,
            fhir_resources={'Patient': fhir_patient, 'Appointment': appointments}
        )
            

    def __call__(self, 
                 data_pair: Tuple[dict, dict], 
                 agent_test_data: dict, 
                 agent_results: dict, 
                 environment: "HospitalEnvironment", 
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
                'test': required_test_list,
                'preference': preference,
            } for preference in gt.get('preference')
        ]

        #################################################### Regular Scheudling Simulation ####################################################
        # Initialize the simulation environment using the first preference data
        staff_known_data = {
            'patient_fv': None,
            'department': None,
            'attending_physician': None,
            'test': None,
            'patient_intention': None,
        }
        preference = gt_data[0].get('preference')
        preference_desc = OPFU_PREFERENCE_PHRASE_PATIENT[preference]
        required_test_desc = [f"{i+1}. {code_to_test_name[test['code']]}" for i, test in enumerate(gt_data[0]['test'])]
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
            },
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
        self._book_schedule(
            status=status,
            prediction=prediction,
            doctor_information=doctor_information,
            test_information=test_information,
            environment=environment
        )
        agent_test_data['doctor'] = doctor_information
        agent_test_data['test'] = test_information

        # Append results
        for key in result_dict.keys():
            results[key] += result_dict[key]
        results['token'].append(self.token_stats)
        #######################################################################################################################################

        # Other events
        ## Simulate the test cancellation requests
        if random.random() < self.schedule_cancellation_prob:
            doctor_information, test_information, result_dict = self.cancellation_request(
                doctor_information=doctor_information,
                test_information=test_information,
                environment=environment,
                verbose=verbose,
            )
            if result_dict is not None:
                agent_test_data['doctor'] = doctor_information
                agent_test_data['test'] = test_information
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

        ## Simulate the rescheduling (move tests earlier) requests
        if random.random() < self.request_early_schedule_prob:
            doctor_information, test_information, result_dict = self.rescheduling_request(
                doctor_information=doctor_information,
                test_information=test_information,
                environment=environment,
                verbose=verbose,
            )
            if result_dict is not None:
                agent_test_data['doctor'] = doctor_information
                agent_test_data['test'] = test_information
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
