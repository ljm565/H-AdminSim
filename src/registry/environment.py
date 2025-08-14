import time
import random
from typing import Union
from copy import deepcopy
from decimal import getcontext

from tasks import FHIRManager
from utils import log
from utils.fhir_utils import (
    get_patient_from_appointment,
    convert_fhir_resources_to_doctor_info,
)
from utils.common_utils import (
    get_iso_time,
    get_utc_offset,
    compare_iso_time,
    exponential_backoff,
    generate_random_iso_time_between,
)



class HospitalEnvironment:
    def __init__(self, config, agent_test_data):
        self.fhir_manager = FHIRManager(config)
        self.__init_variable(agent_test_data)
        
        # Define error codes
        self.status_codes = {
            'format': 'reschedule: incorrect format',
            'information': 'reschedule: information mismatch',
            'physician': 'reschedule: incorrect physician',
            'schedule': 'reschedule: invalid schedule',
            'priority': {'priority': 'reschedule: lower priority', 'booking': 'reschedule: booking priority'},
            'flexibility': 'reschedule: invalid flexibility',
            'status': 'reschedule: invalid status',
            'conflict': 'reschedule: schedule conflict',
            'correct': 'pass',
        }


    def __init_variable(self, agent_test_data: dict):
        """
        Initialize the environment variables based on the agent test data.

        Args:
            agent_test_data (dict): An agent test data to simulate a hospital environmnet.
        """
        getcontext().prec = 10
        self._epsilon = 1e-6
        self.max_retries = 5
        self.HOSPITAL_NAME = agent_test_data.get('metadata').get('hospital_name')
        self._START_HOUR = agent_test_data.get('metadata').get('time').get('start_hour')
        self._END_HOUR = agent_test_data.get('metadata').get('time').get('end_hour')
        self._TIME_UNIT = agent_test_data.get('metadata').get('time').get('interval_hour')
        _country_code = agent_test_data.get('metadata').get('country_code', 'KR')
        
        self._utc_offset = get_utc_offset(_country_code)
        self.current_time = get_iso_time(
            time_hour=random.uniform(max(0, self._START_HOUR - 6), max(0, self._START_HOUR - self._epsilon)),
            utc_offset=self._utc_offset
        )
        self.patient_schedules = list()
        self.first_verbose_flag = True

        # Cache variables
        self._fhir_practitioner_cache = None
        self._fhir_practitionerrole_cache = None
        self._fhir_schedule_cache = None
        self._fhir_slot_cache = None


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
    

    def update_fhir(self, fhir_resources: dict):
        """
        Update resources on the FHIR server.

        fhir_resources (dict): Dictionary where each key is a FHIR resource type (e.g., 'Appointment', 'Slot'),
                               and each value is the corresponding FHIR resource data to be updated.
        """
        # Update the rescheduled appointment FHIR resources
        if fhir_resources['reschedule']:
            appointment_patient_to_id_map = {get_patient_from_appointment(appn['resource']): appn['resource']['id'] for appn in self.fhir_appointment}
            for resource in fhir_resources['reschedule']:
                patient = get_patient_from_appointment(resource)
                self.fhir_manager.delete('Appointment', appointment_patient_to_id_map[patient], verbose=False)

            # Create after delete operation to prevent ID conflicts
            for resource in fhir_resources['reschedule']:
                self.fhir_manager.create('Appointment', resource, verbose=False)
        
        # Create a new appointment
        for resource_type, resource in fhir_resources.items():
            if resource and resource_type.lower() in ['patient', 'appointment']:
                self.fhir_manager.create(resource_type, resource, verbose=False)


    def update_current_time(self):
        """
        Update the current hospital time.
        """
        min_iso_time = self.current_time
        max_iso_time = get_iso_time(self.patient_schedules[-1]['schedule'][-1], utc_offset=self._utc_offset)
        self.current_time = generate_random_iso_time_between(min_iso_time, max_iso_time)

    
    def update_patient_status(self):
        """
        Update the status of each patient based on the current hospital time.
        """
        for schedule in self.patient_schedules:            
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
            
            if len(self.patient_schedules) and patient_schedule['schedule'][0] > self.patient_schedules[-1]['schedule'][0]:
                self.update_current_time()
            
            patient_schedule = deepcopy(patient_schedule)
            del patient_schedule['reschedule']
            self.patient_schedules.append(patient_schedule)
            self.update_patient_status()

        self.reset_variable()
 