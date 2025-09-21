import time
import random
from typing import Union
from decimal import getcontext
from datetime import datetime, timedelta

from tasks import FHIRManager
from utils import log, colorstr
from utils.fhir_utils import convert_fhir_resources_to_doctor_info
from utils.common_utils import (
    get_iso_time,
    get_utc_offset,
    str_to_datetime,
    datetime_to_str,
    compare_iso_time,
    exponential_backoff,
    generate_random_iso_time_between,
)



class HospitalEnvironment:
    def __init__(self, config, agent_test_data):
        # FHIR manager
        self.fhir_manager = FHIRManager(config)
        
        # Basic
        getcontext().prec = 10
        self._epsilon = 1e-6
        self.max_retries = config.fhir_max_connection_retries
        self._days_before = config.booking_days_before_simulation
        self.HOSPITAL_NAME = agent_test_data.get('metadata').get('hospital_name')
        self._START_DATE = agent_test_data.get('metadata').get('start_date')
        self._END_DATE = agent_test_data.get('metadata').get('end_date')
        self._START_HOUR = agent_test_data.get('metadata').get('time').get('start_hour')
        self._END_HOUR = agent_test_data.get('metadata').get('time').get('end_hour')
        self._TIME_UNIT = agent_test_data.get('metadata').get('time').get('interval_hour')
        self._PATIENT_NUM = len(agent_test_data.get('agent_data'))
        _country_code = agent_test_data.get('metadata').get('country_code', 'KR')
        self.booking_num = {k: 0 for k in agent_test_data.get('doctor')}
        
        # Time setting
        self._utc_offset = get_utc_offset(_country_code)
        self.current_time = get_iso_time(
            time_hour=random.uniform(max(0, self._START_HOUR - 6), max(0, self._START_HOUR - self._epsilon)),
            date=datetime_to_str(str_to_datetime(self._START_DATE) - timedelta(days=self._days_before), "%Y-%m-%d"),
            utc_offset=self._utc_offset
        )
        self.avg_gap = self.__calculate_max_time_increment()
        
        # Misc.
        self.patient_schedules = list()
        self.waiting_list = list()
        self.first_verbose_flag = True

        # Cache variables
        self._fhir_practitioner_cache = None
        self._fhir_practitionerrole_cache = None
        self._fhir_schedule_cache = None
        self._fhir_slot_cache = None


    def __calculate_max_time_increment(self) -> float:
        """
        Calculate the maximum average time increment (gap) between patient booking within the defined scheduling period.

        Returns:
            float: The average time gap (in hours) between patients.
        """
        st = str_to_datetime(get_iso_time(self._START_HOUR, self._START_DATE, self._utc_offset))
        tr = str_to_datetime(get_iso_time(self._END_HOUR, self._END_DATE, self._utc_offset))
        total_hours = (tr - st).total_seconds() / 3600
        avg_gap = total_hours / self._PATIENT_NUM
        return avg_gap


    def doctor_info_from_fhir(self, use_cache: bool = True) -> dict:
        """
        Build a doctor information dictionary from FHIR resources for simulation.

        Args:
            use_cache (bool): If True, reuse cached FHIR resources if available. Defaults to True.

        Returns:
            dict: doctor_information (dict): Dictionary of doctor data including their existing schedules.
                                             Each key is a doctor's name, and each value includes a 'schedule' field.
        """
        if self.first_verbose_flag:
            log('Build doctor information from the FHIR resources..')
            self.first_verbose_flag = False

        hospital_id = self.HOSPITAL_NAME.replace('_', '')
        cache_ready = all([
            self._fhir_practitioner_cache,
            self._fhir_practitionerrole_cache,
            self._fhir_schedule_cache,
            self._fhir_slot_cache,
        ])
        
        if not use_cache or not cache_ready:
            self._fhir_practitioner_cache = [
                x for x in self.fhir_manager.read_all('Practitioner', verbose=False)
                if hospital_id in x['resource']['id']
            ]
            self._fhir_practitionerrole_cache = [
                x for x in self.fhir_manager.read_all('PractitionerRole', verbose=False)
                if hospital_id in x['resource']['id']
            ]
            self._fhir_schedule_cache = [
                x for x in self.fhir_manager.read_all('Schedule', verbose=False)
                if hospital_id in x['resource']['id']
            ]
            self._fhir_slot_cache = [
                x for x in self.fhir_manager.read_all('Slot', verbose=False)
                if hospital_id in x['resource']['id']
            ]

        # Get Appointment resources from the FHIR server
        # NOTE: Sometimes, a FHIR resource is accessed before it gets updated, so the operation is performed with a retry flag 
        retry_count = 0
        while 1:
            try:
                self.fhir_appointment = [
                    x for x in self.fhir_manager.read_all('Appointment', verbose=False)
                    if hospital_id in x['resource']['id']
                ]
                assert len(self.fhir_appointment) == len(self.patient_schedules), f"Mismatch in appointment count: expected {len(self.patient_schedules)}, got {len(self.fhir_appointment)}"
                break
            except AssertionError as e:
                if retry_count >= self.max_retries:
                    log(f"\nMax retries reached. Last error: {e}", level='error')
                    raise e
                wait_time = exponential_backoff(retry_count)
                log(f"[{retry_count + 1}/{self.max_retries}] {type(e).__name__}: {e}. Retrying in {wait_time:.1f} seconds...", level='warning')
                time.sleep(wait_time)
                retry_count += 1
                continue

        # Convert resources regardless of whether they came from cache or fresh read
        doctor_information = convert_fhir_resources_to_doctor_info(
            self._fhir_practitioner_cache,
            self._fhir_practitionerrole_cache,
            self._fhir_schedule_cache,
            self._fhir_slot_cache,
            self.fhir_appointment,
            **{'start': self._START_HOUR, 'end': self._END_HOUR, 'interval': self._TIME_UNIT}
        )
        return doctor_information
    

    def resume(self, agent_results: dict):
        """
        Resume the hospital environment from previously saved agent results.

        Args:
            agent_test_data (dict): Input data containing static information 
                                    about doctors, patients, and other hospital resources.
            agent_results (dict): Previously saved results from the agent's simulation.
        """
        if 'schedule' in agent_results:
            for status, pred in zip(agent_results['schedule']['status'], agent_results['schedule']['pred']):
                if status:
                    self.patient_schedules.append(pred)
                    self.booking_num[pred['attending_physician']] += 1
                    self.current_time = pred['last_updated_time']
            
            self.update_current_time()
            self.update_patient_status()
    

    def schedule_cancel_event(self, idx: int, verbose: bool = False):
        """
        Cancel a scheduled event for the patient.
        This method updates the status of a scheduled event at the given index
        to 'cancelled'. Index values greater than or equal to 0 are allowed.

        Args:
            idx (int): The index of the schedule to cancel. Must be 0 or a positive integer.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.
        """
        if idx >= 0:
            for turn, (_, schedule) in enumerate(self.waiting_list):
                if schedule == self.patient_schedules[idx]:
                    self.pop_waiting_list(turn, verbose)
                    break
            self.patient_schedules[idx]['status'] = 'cancelled'
            if verbose:
                log(f'{colorstr("CANCELED")}: {self.patient_schedules[idx]} schedule is canceled.')
    

    def add_waiting_list(self, idx: int, verbose: bool = False):
        """
        Add a schedule to the waiting list for an earlier appointment if needed.

        Args:
            idx (int): The index of the schedule to add to the waiting list. Must be 0 or a positive integer.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.
        """
        if idx >= 0:
            requested_schedule = self.patient_schedules[idx]
            if all(requested_schedule != s[1] for s in self.waiting_list):
                self.waiting_list.append((idx, requested_schedule))
                if verbose:
                    log(f'{colorstr("WAITING LIST")}: {requested_schedule} schedule is appended to the waiting list.')

    
    def pop_waiting_list(self, idx: Union[list[int], int], verbose: bool = False):
        """
        Pop a schedule to the waiting list.

        Args:
            idx (Union[list[int], int]): The index list (or index) of the schedule to pop from the waiting list.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.
        """
        if isinstance(idx, int) and idx >= 0:
            idx = [idx]

        if len(idx):
            idx = sorted(idx, reverse=True)
            for _id in idx:
                schedule = self.waiting_list.pop(_id)
                if verbose:
                    log(f'{colorstr("WAITING LIST")}: {schedule[1]} schedule is popped from the waiting list.')


    def update_fhir(self, fhir_resources: dict):
        """
        Update resources on the FHIR server.

        fhir_resources (dict): Dictionary where each key is a FHIR resource type (e.g., 'Appointment', 'Slot'),
                               and each value is the corresponding FHIR resource data to be updated.
        """
        # Update new FHIR resources
        for resource_type, resource in fhir_resources.items():
            if resource and resource_type.lower() in ['patient', 'appointment']:
                self.fhir_manager.create(resource_type, resource, verbose=False)
                

    def delete_fhir(self, fhir_resources: dict):
        """
        Delete resources on the FHIR server.

        Args:
            fhir_resources (dict): Dictionary where each key is a FHIR resource type (e.g., 'Appointment', 'Slot'),
                                   and each value is the corresponding FHIR resource data to be updated.
        """
        # Delete the existing FHIR resources
        for resource_type, resource in fhir_resources.items():
            if resource and resource_type.lower() in ['patient', 'appointment']:
                self.fhir_manager.delete(resource_type, resource['id'], verbose=False)
    

    def update_current_time(self):
        """
        Update the current hospital time.
        """
        min_iso_time = self.current_time
        max_iso_time = (str_to_datetime(self.current_time) + timedelta(hours=self.avg_gap)).isoformat(timespec='seconds')
        self.current_time = generate_random_iso_time_between(min_iso_time, max_iso_time)

    
    def update_patient_status(self):
        """
        Update the status of each patient based on the current hospital time.
        """
        for schedule in self.patient_schedules:            
            if schedule.get('status') == 'cancelled':
                continue

            tmp_st_iso_time = get_iso_time(schedule['schedule'][0], utc_offset=self._utc_offset)
            tmp_tr_iso_time = get_iso_time(schedule['schedule'][-1], utc_offset=self._utc_offset)

            if compare_iso_time(self.current_time, tmp_tr_iso_time):
                status = 'completed'
            elif compare_iso_time(tmp_st_iso_time, self.current_time):
                status = 'scheduled'
            else: 
                status = 'in_progress'
            
            schedule['status'] = status


    def reset_variable(self):
        """
        Reset variables.
        """
        pass

    
    def update_env(self, 
                   status: bool,
                   patient_schedule: Union[dict, str],
                   fhir_resources: dict):
        """
        Update the hospital environment after successfully assigning an appointment.

        Args:
            status (bool): Whether the appointment was successfully assigned.
            patient_schedule (Union[dict, str]): The patient's new schedule to add. Should contain a 'schedule' key with start and end time.
            fhir_resources (dict): Dictionary where each key is a FHIR resource type (e.g., 'Appointment', 'Slot'),
                                   and each value is the corresponding FHIR resource data to be updated.
        """
        if status:
            self.update_fhir(fhir_resources)
            self.update_current_time()
            self.patient_schedules.append(patient_schedule)
            self.update_patient_status()
            self.booking_num[patient_schedule['attending_physician']] += 1

        self.reset_variable()
 