import time
import random
from copy import deepcopy
from decimal import getcontext
from datetime import timedelta
from typing import Union, Tuple, Optional

from h_adminsim.task.fhir_manager import FHIRManager
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.fhir_utils import get_all_doctor_info
from h_adminsim.utils.common_utils import (
    iso_to_date,
    iso_to_hour,
    get_iso_time,
    sort_schedule,
    get_utc_offset,
    str_to_datetime,
    datetime_to_str,
    compare_iso_time,
    exponential_backoff,
    generate_random_iso_time_between,
    convert_time_list_to_merged_time,
)



class HospitalEnvironment:
    def __init__(self, 
                 agent_test_data: dict,
                 fhir_url: Optional[str] = None,
                 fhir_max_connection_retries: int = 5,
                 start_day_before: float = 3):
        
        # FHIR manager
        self.fhir_manager = FHIRManager(fhir_url) if fhir_url else None
        
        # Basic
        getcontext().prec = 10
        self._epsilon = 1e-6
        self.max_retries = fhir_max_connection_retries
        self._days_before = start_day_before
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


    def get_general_doctor_info_from_fhir(self, use_cache: bool = True) -> dict:
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
                valid_len = len(list(filter(lambda x: x['status'] != 'cancelled', self.patient_schedules)))
                assert len(self.fhir_appointment) == valid_len, f"Mismatch in appointment count: expected {valid_len}, got {len(self.fhir_appointment)}"
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
        doctor_information = get_all_doctor_info(
            self._fhir_practitioner_cache,
            self._fhir_practitionerrole_cache,
            self._fhir_schedule_cache,
            self._fhir_slot_cache,
            self.fhir_appointment,
            **{'start': self._START_HOUR, 'end': self._END_HOUR, 'interval': self._TIME_UNIT}
        )
        return doctor_information
    

    def get_doctor_schedule(self,
                            doctor_information: Optional[dict] = None,
                            *,
                            department: Optional[str] = None,
                            fhir_integration: bool = False,
                            express_detail: bool = False) -> dict:
        """
        Build doctor schedules for a given department.

        Args:
            doctor_information (Optional[dict], optional): Simulation doctor data (used when fhir_integration is False). Defaults to None.
            department (Optional[str], optional): Target department name. Defaults to None.
            fhir_integration (bool, optional): If True, build schedules from FHIR resources. Defaults to False.
            express_detail (bool, optional): If True, express schedules with explicit start/end fields. Defualtsto False.

        Returns:
            dict: Filtered doctor scheduling information.
        """
        def __build_single_doctor_schedule(practitioner_role: dict) -> Tuple[str, dict]:
            """
            Build scheduling information for a single doctor.

            Args:
                practitioner_role (dict): A FHIR PractitionerRole resource for a doctor, already filtered by department.

            Returns:
                Tuple[str, dict]: Doctor name and his (or her) information dictionary containing the constructed scheduling information.
            """
            schedule, appointments = dict(), list()
            practitioner_id = practitioner_role['practitioner']['reference']
            practitioner = self.fhir_manager.read('Practitioner', practitioner_id.split('/')[-1], verbose=False).json()
            practitioner_schedule_id = self.fhir_manager.read_all('Schedule', params={'actor': practitioner_id}, verbose=False)[0]['resource']['id']
            fixed_slots = [slot['resource'] for slot in self.fhir_manager.read_all('Slot', params={'schedule': f'Schedule/{practitioner_schedule_id}'}, verbose=False)]

            # Append fixed schedules of a doctor
            for slot in fixed_slots:
                date = iso_to_date(slot['start'])
                schedule.setdefault(date, [])
                if slot['status'] != 'free':
                    schedule[date].append([iso_to_hour(slot['start']), iso_to_hour(slot['end'])])
                
                # Get all appointments related to this slot
                appointment_resources = self.fhir_manager.read_all('Appointment', params={'slot': f'Slot/{slot["id"]}'}, verbose=False)
                if len(appointment_resources) > 0:
                    appointments.append(appointment_resources[0]['resource'])
            
            # Merge fixed schedule times
            for date, time_list in schedule.items():
                schedule[date] = convert_time_list_to_merged_time(
                    time_list=sort_schedule(time_list),
                    start=self._START_HOUR,
                    end=self._END_HOUR,
                    interval=self._TIME_UNIT
                )

            # Append patient appointments of a doctor
            for appointment in appointments:
                date = iso_to_date(appointment['start'])
                schedule.setdefault(date, [])
                schedule[date].append([iso_to_hour(appointment['start']), iso_to_hour(appointment['end'])])            
            
            # Collect doctor's information
            name = f"{practitioner['name'][0]['prefix'][0]} {practitioner['name'][0]['given'][0]} {practitioner['name'][0]['family']}"
            department = practitioner_role['specialty'][0]['text']
            specialty = {
                'name': practitioner_role['specialty'][0]['coding'][0]['display'],
                'code': practitioner_role['specialty'][0]['coding'][0]['code']
            }
            capacity_attributes = {attr['text']: attr['coding'][0]['display'] for attr in practitioner_role['characteristic']}
            workload = f"{round(self.booking_num[name] / int(capacity_attributes['capacity']) * 100, 2)}%"
            outpatient_duration = 1 / int(capacity_attributes['capacity_per_hour'])
            information = {
                'department': department,
                'specialty': specialty,
                'schedule': sort_schedule(schedule),
                'workload': workload,
                'outpatient_duration': outpatient_duration
            } 
            return name, information

        filtered_doctor_information = {'doctor': {}}

        # Get filtered doctor information directly from FHIR
        if fhir_integration:
            if self.first_verbose_flag:
                log('Build doctor information from the FHIR resources..')
                self.first_verbose_flag = False
            
            # Get doctors belonging to the department
            params={"specialty:text": department}
            practitioner_roles = [resource['resource'] for resource in self.fhir_manager.read_all('PractitionerRole', params=params, verbose=False)]

            for practitioner_role in practitioner_roles:
                doctor_name, doctor_schedule = __build_single_doctor_schedule(practitioner_role)
                filtered_doctor_information['doctor'][doctor_name] = doctor_schedule
        
        # Get filtered doctor information from the simulation data
        else:
            for k, v in doctor_information.items():
                if v['department'] == department:
                    tmp_schedule = deepcopy(v)
                    del tmp_schedule['capacity_per_hour'], tmp_schedule['capacity'], tmp_schedule['gender'], tmp_schedule['telecom'], tmp_schedule['birthDate']
                    tmp_schedule['workload'] = f"{round(self.booking_num[k] / v['capacity'] * 100, 2)}%"
                    tmp_schedule['outpatient_duration'] = 1 / v['capacity_per_hour']
                    filtered_doctor_information['doctor'][k] = tmp_schedule


        # Whether express more details in the built schedules
        if express_detail:
            for _, info in filtered_doctor_information['doctor'].items():
                info['schedule'] = {
                    date: [{'start': s[0], 'end': s[1]} for s in schedule]
                    for date, schedule in info['schedule'].items()
                }
        
        return filtered_doctor_information
    

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
                    if not pred['status'] == 'cancelled':
                        self.booking_num[pred['attending_physician']] += 1
                    self.current_time = pred['last_updated_time']
            
            self.waiting_list = sorted([(i, s) for i, s in enumerate(self.patient_schedules) if s['waiting_order'] >= 0], key=lambda x: x[1]['waiting_order'])
            
            log(f"Resumed hospital time set to {self.current_time}.")
            log(f"Resumed hospital environment with {len(self.patient_schedules)} patient schedules.")
            log(f"Resumed waiting list with {len(self.waiting_list)} patient schedules.")
            log(f"Current booking numbers per doctor: {self.booking_num}")

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
            for turn, (i, _) in enumerate(self.waiting_list):
                if i == idx:
                    self.pop_waiting_list(turn, verbose)
                    break
            self.patient_schedules[idx]['status'] = 'cancelled'
            self.patient_schedules[idx]['last_updated_time'] = self.current_time
            self.booking_num[self.patient_schedules[idx]['attending_physician']] -= 1
            if verbose:
                log(f'{colorstr("[CANCELLED]")}: {self.patient_schedules[idx]} schedule is cancelled.')
    

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
                requested_schedule['waiting_order'] = len(self.waiting_list)
                self.waiting_list.append((idx, requested_schedule))
                if verbose:
                    log(f'{colorstr("[WAITING LIST ADDED]")}: {requested_schedule} schedule is appended to the waiting list.')

    
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
                schedule[1]['waiting_order'] = -1
                if verbose:
                    log(f'{colorstr("[WAITING LIST POPPED]")}: {schedule[1]} schedule is popped from the waiting list.')
        
            for i, (_, schedule) in enumerate(self.waiting_list):
                schedule['waiting_order'] = i


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
            if schedule.get('waiting_order', -1) < 0:
                schedule['waiting_order'] = -1

            if schedule.get('status') == 'cancelled':
                continue

            tmp_st_iso_time = get_iso_time(schedule['schedule'][0], date=schedule['date'], utc_offset=self._utc_offset)
            tmp_tr_iso_time = get_iso_time(schedule['schedule'][-1], date=schedule['date'], utc_offset=self._utc_offset)

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
 