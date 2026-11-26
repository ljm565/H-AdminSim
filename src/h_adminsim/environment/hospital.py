import time
import random
from copy import deepcopy
from datetime import timedelta
from typing import Union, Optional
from collections import defaultdict
from decimal import Decimal, getcontext

from h_adminsim.task.fhir_manager import FHIRManager
from h_adminsim.registry.variables import SCHEDULE_STATUS
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.fhir_utils import (
    get_all_test_info,
    get_all_doctor_info,
)
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
        """
        Hospital state for a simulation: current time, patient schedules, waiting list, test capacity, FHIR cache.

        Args:
            agent_test_data (dict): One hospital's agent data (``metadata``, ``doctor``, ``test``, ``agent_data``).
            fhir_url (Optional[str], optional): FHIR server URL; None disables FHIR. Defaults to None.
            fhir_max_connection_retries (int, optional): Retries when FHIR appointments lag behind the local state. Defaults to 5.
            start_day_before (float, optional): Days before the start date at which the clock starts. Defaults to 3.
        """
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
        self._DOCTOR_SCHEDULE_START_DATE = agent_test_data.get('metadata').get('doctor_schedule_start_date')
        self._START_HOUR = agent_test_data.get('metadata').get('time').get('start_hour')
        self._END_HOUR = agent_test_data.get('metadata').get('time').get('end_hour')
        self._TIME_UNIT = agent_test_data.get('metadata').get('time').get('interval_hour')
        self._PATIENT_NUM = len(agent_test_data.get('agent_data')) if agent_test_data.get('agent_data', None) != None else 100
        _country_code = agent_test_data.get('metadata').get('country_code', 'KR')
        self.booking_num = {k: 0 for k in agent_test_data.get('doctor')}
        
        # Time setting
        self._utc_offset = get_utc_offset(_country_code)
        self.current_time = get_iso_time(
            time_hour=random.uniform(max(0, self._START_HOUR - 6), max(0, self._START_HOUR - self._epsilon)),
            date=datetime_to_str(str_to_datetime(self._START_DATE) - timedelta(days=self._days_before), "%Y-%m-%d"),
            utc_offset=self._utc_offset
        )
        self.avg_gap = self._calculate_max_time_increment()

        # Total capacity and fixed (pre-booked) hours per (test, date)
        self.test_capacity, self._test_fixed_booked, self.test_duration = self._calculate_test_capacity(agent_test_data.get('test') or {})

        # Misc.
        self.patient_schedules = list()
        self.waiting_list = list()
        self.doctor_first_verbose_flag = True
        self.test_first_verbose_flag = True

        # Cache variables
        self._fhir_practitioner_cache = None
        self._fhir_practitionerrole_cache = None
        self._fhir_healthcareservice_cache = None
        self._fhir_device_cache = None
        self._fhir_schedule_cache = None
        self._fhir_slot_cache = None


    def _calculate_max_time_increment(self) -> float:
        """
        Average gap between patient bookings: the scheduling period divided by the number of patients.

        Returns:
            float: The average gap in hours.
        """
        st = str_to_datetime(get_iso_time(self._START_HOUR, self._START_DATE, self._utc_offset))
        tr = str_to_datetime(get_iso_time(self._END_HOUR, self._END_DATE, self._utc_offset))
        total_hours = (tr - st).total_seconds() / 3600
        avg_gap = total_hours / self._PATIENT_NUM
        return avg_gap


    def _calculate_test_capacity(self, test_information: dict) -> tuple[dict, dict, dict]:
        """
        Working hours and fixed (pre-booked) hours of every test per date, pooled over its devices.
        A fully booked device-date is a closed day and counts toward neither.

        Args:
            test_information (dict): Static test data `{department: [test_dict, ...]}` holding fixed device schedules.

        Returns:
            tuple[dict, dict, dict]: `{(test_code, date): capacity_hours}`, `{(test_code, date): fixed_booked_hours}` (`Decimal`), `{test_code: duration_hour}`.
        """
        capacity, fixed, duration = defaultdict(Decimal), defaultdict(Decimal), {}
        day_hours = Decimal(str(self._END_HOUR)) - Decimal(str(self._START_HOUR))
        for tests in test_information.values():
            for test in tests:
                duration[test['code']] = test['duration_hour']
                for device_info in test.get('devices', {}).values():
                    for date, schedules in device_info.get('schedule', {}).items():
                        busy = sum((Decimal(str(e)) - Decimal(str(s)) for s, e in schedules), Decimal(0))
                        if busy >= day_hours:
                            capacity[(test['code'], date)] += 0     # Closed day: keep the key with zero capacity
                            continue
                        capacity[(test['code'], date)] += day_hours
                        fixed[(test['code'], date)] += busy
        return {k: float(v) for k, v in capacity.items()}, dict(fixed), duration


    def booked_test_hours(self, exclude: Optional[dict] = None) -> dict:
        """
        Test hours already booked per (test, date): fixed bookings plus non-cancelled follow-up bookings of any urgency.
        Read from `patient_schedules`, so cancellations, reschedules, and resumes are reflected.

        Args:
            exclude (Optional[dict], optional): A booking to leave out (e.g. the one being rescheduled). Defaults to None.

        Returns:
            dict: `{(test_code, date): booked_hours}`.
        """
        used = defaultdict(Decimal, self._test_fixed_booked)
        for schedule in self.patient_schedules:
            if schedule is exclude or schedule.get('visit_type') != 'follow_up_visit':
                continue
            if schedule.get('status') == SCHEDULE_STATUS['cancelled']:
                continue
            for test in schedule.get('test') or []:
                if test.get('status') == SCHEDULE_STATUS['cancelled'] or not test.get('schedule'):
                    continue
                used[(test['code'], test['date'])] += Decimal(str(test['schedule'][1])) - Decimal(str(test['schedule'][0]))
        return {k: float(v) for k, v in used.items()}


    def get_general_doctor_info_from_fhir(self, use_cache: bool = True) -> dict:
        """
        Build a doctor information dictionary from FHIR resources for simulation.

        Args:
            use_cache (bool, optional): Reuse cached FHIR resources (appointments are always re-read). Defaults to True.

        Returns:
            dict: ``{doctor name: info with 'schedule'}``.
        """
        if self.doctor_first_verbose_flag:
            log('Build doctor information from the FHIR resources..')
            self.doctor_first_verbose_flag = False

        hospital_id = self.HOSPITAL_NAME.replace('_', '')
        
        if not use_cache or not self._fhir_practitioner_cache:
            self._fhir_practitioner_cache = [
                x for x in self.fhir_manager.read_all('Practitioner', verbose=False)
                if hospital_id in x['resource']['id']
            ]

        if not use_cache or not self._fhir_practitionerrole_cache:
            self._fhir_practitionerrole_cache = [
                x for x in self.fhir_manager.read_all('PractitionerRole', verbose=False)
                if hospital_id in x['resource']['id']
            ]
        
        if not use_cache or not self._fhir_schedule_cache:
            self._fhir_schedule_cache = [
                x for x in self.fhir_manager.read_all('Schedule', verbose=False)
                if hospital_id in x['resource']['id']
            ]
        
        if not use_cache or not self._fhir_slot_cache:
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
                    if hospital_id in x['resource']['id'] and 'Dr.' in x['resource']['id']
                ]
                valid_len = len(list(filter(lambda x: x['schedule'] is not None and x['status'] != SCHEDULE_STATUS['cancelled'], self.patient_schedules)))
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
    

    def get_general_test_info_from_fhir(self, use_cache: bool = True) -> dict:
        """
        Build a test information dictionary from FHIR resources for simulation.

        Args:
            use_cache (bool, optional): Reuse cached FHIR resources (appointments are always re-read). Defaults to True.

        Returns:
            dict: ``{department: [test info with device schedules]}``.
        """
        if self.test_first_verbose_flag:
            log('Build test information from the FHIR resources..')
            self.test_first_verbose_flag = False

        hospital_id = self.HOSPITAL_NAME.replace('_', '')
        
        if not use_cache or not self._fhir_healthcareservice_cache:
            self._fhir_healthcareservice_cache = [
                x for x in self.fhir_manager.read_all('HealthcareService', verbose=False)
                if hospital_id in x['resource']['id']
            ]
        
        if not use_cache or not self._fhir_device_cache:
            self._fhir_device_cache = [
                x for x in self.fhir_manager.read_all('Device', verbose=False)
                if hospital_id in x['resource']['id']
            ]
        
        if not use_cache or not self._fhir_schedule_cache:
            self._fhir_schedule_cache = [
                x for x in self.fhir_manager.read_all('Schedule', verbose=False)
                if hospital_id in x['resource']['id']
            ]
        
        if not use_cache or not self._fhir_slot_cache:
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
                    if hospital_id in x['resource']['id'] and 'Dr.' not in x['resource']['id']
                ]
                valid_len = 0
                for appn in self.patient_schedules:
                    if appn['visit_type'] == 'follow_up_visit':
                        for t in appn.get('test') or []:
                            if t.get('status') != SCHEDULE_STATUS['cancelled']:
                                valid_len += 1
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
        test_information = get_all_test_info(
            self._fhir_practitionerrole_cache,
            self._fhir_healthcareservice_cache,
            self._fhir_device_cache,
            self._fhir_schedule_cache,
            self._fhir_slot_cache,
            self.fhir_appointment,
            **{'start': self._START_HOUR, 'end': self._END_HOUR, 'interval': self._TIME_UNIT}
        )
        return test_information
    

    def get_doctor_schedule(self,
                            department: str,
                            doctor_information: Optional[dict] = None,
                            *,
                            fhir_integration: bool = False,
                            express_detail: bool = False) -> dict:
        """
        Build doctor schedules for a given department.

        Args:
            department (str): Target department name.
            doctor_information (Optional[dict], optional): Simulation doctor data (used when fhir_integration is False). Defaults to None.
            fhir_integration (bool, optional): If True, build schedules from FHIR resources. Defaults to False.
            express_detail (bool, optional): If True, express schedules with explicit start/end fields. Defaults to False.

        Returns:
            dict: ``{'doctor': {doctor name: {'department', 'specialty', 'schedule', 'workload', 'outpatient_duration'}}}``.
        """
        def _build_single_doctor_schedule(practitioner_role: dict) -> tuple[str, dict]:
            """
            Build scheduling information for a single doctor.

            Args:
                practitioner_role (dict): A FHIR PractitionerRole resource for a doctor, already filtered by department.

            Returns:
                tuple[str, dict]: Doctor name and his (or her) information dictionary containing the constructed scheduling information.
            """
            schedule, appointments = dict(), list()
            practitioner_id = practitioner_role['practitioner']['reference']
            practitioner = self.fhir_manager.read('Practitioner', practitioner_id.split('/')[-1], verbose=False).json()
            practitioner_schedule_id = self.fhir_manager.read_all('Schedule', params={'actor': practitioner_id}, verbose=False)[0]['resource']['id']
            fixed_slots = [slot['resource'] for slot in self.fhir_manager.read_all('Slot', params={'schedule': f'Schedule/{practitioner_schedule_id}'}, verbose=False)]

            # Append fixed schedules of a doctor
            seen_appointment_ids = set()
            for slot in fixed_slots:
                date = iso_to_date(slot['start'])
                schedule.setdefault(date, [])
                if slot['status'] != 'free':
                    schedule[date].append([iso_to_hour(slot['start']), iso_to_hour(slot['end'])])

                # Collect each related appointment at most once
                appointment_resources = self.fhir_manager.read_all('Appointment', params={'slot': f'Slot/{slot["id"]}'}, verbose=False)
                if len(appointment_resources) > 0:
                    appt = appointment_resources[0]['resource']
                    if appt['id'] not in seen_appointment_ids:
                        appointments.append(appt)
                        seen_appointment_ids.add(appt['id'])
            
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
            if self.doctor_first_verbose_flag:
                log('Build doctor information from the FHIR resources..')
                self.doctor_first_verbose_flag = False
            
            # Get doctors belonging to the department
            hospital_id = self.HOSPITAL_NAME.replace('_', '')
            params = {"specialty:text": department}
            practitioner_roles = [resource['resource'] for resource in self.fhir_manager.read_all('PractitionerRole', params=params, verbose=False)
                                  if hospital_id in resource['resource']['id']]

            for practitioner_role in practitioner_roles:
                doctor_name, doctor_schedule = _build_single_doctor_schedule(practitioner_role)
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
    

    def get_test_device_schedule(self,
                                 test_code: list[str],
                                 test_device_information: Optional[dict] = None,
                                 *,
                                 fhir_integration: bool = False,
                                 express_detail: bool = False) -> dict:
        """
        Build test device schedules for the given test codes.

        Args:
            test_code (list[str]): Test codes the patient must take.
            test_device_information (Optional[dict], optional): Simulation test data ``{department: [test_dict, ...]}`` (used when ``fhir_integration`` is False). Defaults to None.
            fhir_integration (bool, optional): If True, build schedules from FHIR resources. Defaults to False.
            express_detail (bool, optional): If True, express device schedules with explicit ``start``/``end`` fields. Defaults to False.

        Returns:
            dict: ``{'test': {test_code: {..., 'devices': {device_name: {'schedule': {...}}}}}}``.
        """
        filtered_test_device_information = {'test': {}}

        # Get filtered test information directly from FHIR
        if fhir_integration:
            all_test_information = self.get_general_test_info_from_fhir()
            for tests in all_test_information.values():
                for test in tests:
                    if test['code'] in test_code:
                        filtered_test_device_information['test'][test['code']] = deepcopy(test)

        # Get filtered test information from the simulation data
        else:
            for tests in test_device_information.values():
                for test in tests:
                    if test['code'] in test_code:
                        filtered_test_device_information['test'][test['code']] = deepcopy(test)

        # Whether express more details in the built schedules
        if express_detail:
            for info in filtered_test_device_information['test'].values():
                for device_info in info.get('devices', {}).values():
                    device_info['schedule'] = {
                        date: [{'start': s[0], 'end': s[1]} for s in schedule]
                        for date, schedule in device_info.get('schedule', {}).items()
                    }

        return filtered_test_device_information
    

    def resume(self, agent_results: dict, virtual_first_visits: Optional[list] = None) -> None:
        """
        Restore schedules, booking counts, waiting list, and time from previously saved agent results.

        Args:
            agent_results (dict): Previously saved results of the simulation.
            virtual_first_visits (Optional[list], optional): Virtual first-visit bookings of the previous run (not in `agent_results`). Defaults to None.
        """
        new_time = None

        # Seeded past visits come first
        for virtual in virtual_first_visits or []:
            self.patient_schedules.append(virtual)
            self.booking_num[virtual['attending_physician']] += 1

        for task_name in ('first_visit_scheduling', 'follow_up_visit_scheduling'):
            if task_name not in agent_results:
                continue

            statuses = [x for y in agent_results[task_name]['status'] for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
            preds = [x for y in agent_results[task_name]['pred'] for x in (y if isinstance(y, list) or isinstance(y, tuple) else [y])]
            for status, pred in zip(statuses, preds):
                if not (isinstance(status, bool) and status):
                    continue

                # Retrieval-only records (a cancel/reschedule tool's identified index) carry no booking
                if not isinstance(pred, dict) or 'patient' not in pred:
                    continue

                # Mirror update_env: the booking is recorded even when its consultation slot is empty
                self.patient_schedules.append(pred)
                if new_time is None or compare_iso_time(pred['last_updated_time'], new_time):
                    new_time = pred['last_updated_time']

                if pred.get('schedule') and pred.get('status') != SCHEDULE_STATUS['cancelled']:
                    self.booking_num[pred['attending_physician']] += 1

        # Waiting list update
        self.waiting_list = sorted(
            [(i, s) for i, s in enumerate(self.patient_schedules) if s.get('waiting_order', -1) >= 0],
            key=lambda x: x[1]['waiting_order'],
        )

        if new_time:
            self.update_current_time(new_time)

        log(f"Resumed hospital time set to {self.current_time}.")
        log(f"Resumed hospital environment with {len(self.patient_schedules)} patient schedules ({len(virtual_first_visits or [])} virtual first-visit(s)).")
        log(f"Resumed waiting list with {len(self.waiting_list)} patient schedules.")
        log(f"Current booking numbers per doctor: {self.booking_num}")

    

    def schedule_cancel_event(self, idx: int, verbose: bool = False) -> None:
        """
        Cancel the patient schedule at ``idx`` (removing it from the waiting list); negative indices are ignored.

        Args:
            idx (int): Index into ``patient_schedules``.
            verbose (bool, optional): Whether to log the result. Defaults to False.
        """
        if idx >= 0:
            for turn, (i, _) in enumerate(self.waiting_list):
                if i == idx:
                    self._pop_waiting_list(turn, verbose)
                    break
            self.patient_schedules[idx]['status'] = SCHEDULE_STATUS['cancelled']
            self.patient_schedules[idx]['last_updated_time'] = self.current_time
            self.booking_num[self.patient_schedules[idx]['attending_physician']] -= 1
            if verbose:
                log(f'{colorstr("[CANCELLED]")}: {self.patient_schedules[idx]} schedule is cancelled.')
    

    def add_waiting_list(self, idx: int, verbose: bool = False) -> None:
        """
        Add a schedule to the waiting list for an earlier appointment, unless already there; negative indices are ignored.

        Args:
            idx (int): Index into ``patient_schedules``.
            verbose (bool, optional): Whether to log the result. Defaults to False.
        """
        if idx >= 0:
            requested_schedule = self.patient_schedules[idx]
            if all(requested_schedule != s[1] for s in self.waiting_list):
                requested_schedule['waiting_order'] = len(self.waiting_list)
                self.waiting_list.append((idx, requested_schedule))
                if verbose:
                    log(f'{colorstr("[WAITING LIST ADDED]")}: {requested_schedule} schedule is appended to the waiting list.')

    
    def _pop_waiting_list(self, idx: Union[list[int], int], verbose: bool = False) -> None:
        """
        Remove entries from the waiting list and renumber the rest.

        Args:
            idx (Union[list[int], int]): Waiting-list position(s) to remove.
            verbose (bool, optional): Whether to log the result. Defaults to False.
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


    def update_fhir(self, fhir_resources: dict) -> None:
        """
        Create the Patient and Appointment resources on the FHIR server; other types are skipped.

        Args:
            fhir_resources (dict): ``{resource type: resource or list of resources}``.
        """
        # Update new FHIR resources
        for resource_type, resource in fhir_resources.items():
            if not resource or resource_type.lower() not in ['patient', 'appointment']:
                continue
            items = resource if isinstance(resource, list) else [resource]
            for item in items:
                if item:
                    self.fhir_manager.create(resource_type, item, verbose=False)
                

    def delete_fhir(self, fhir_resources: dict) -> None:
        """
        Delete the Patient and Appointment resources from the FHIR server; other types are skipped.

        Args:
            fhir_resources (dict): ``{resource type: resource}``.
        """
        # Delete the existing FHIR resources
        for resource_type, resource in fhir_resources.items():
            if resource and resource_type.lower() in ['patient', 'appointment']:
                self.fhir_manager.delete(resource_type, resource['id'], verbose=False)
    

    def update_current_time(self, new_time: Optional[str] = None) -> None:
        """
        Advance the hospital time and refresh patient statuses.

        Args:
            new_time (Optional[str], optional): New ISO time; None advances by a random amount up to ``avg_gap`` hours. Defaults to None.
        """
        if new_time is None:
            min_iso_time = self.current_time
            max_iso_time = (str_to_datetime(self.current_time) + timedelta(hours=self.avg_gap)).isoformat(timespec='seconds')
            self.current_time = generate_random_iso_time_between(min_iso_time, max_iso_time)
            self.update_patient_status()
        else:
            self.current_time = new_time
            self.update_patient_status()

    
    def update_patient_status(self) -> None:
        """
        Update the status of each non-cancelled consultation and test from the current hospital time.
        """
        def _update_status(st_iso_time: Optional[str] = None, 
                           tr_iso_time: Optional[str] = None,
                           is_not_yet: Optional[bool] = False) -> str:
            """
            Update the status.

            Args:
                st_iso_time (Optional[str], optional): Start time of the schedule in ISO format. Defaults to None.
                tr_iso_time (Optional[str], optional): End time of the schedule in ISO format. Defaults to None.
                is_not_yet (Optional[bool], optional): Flag indicating if the schedule is not yet started. Defaults to False.

            Returns:
                str: The updated status of the schedule.
            """
            if is_not_yet:
                return SCHEDULE_STATUS['not_yet']
            else:
                if compare_iso_time(self.current_time, tr_iso_time):
                    return SCHEDULE_STATUS['completed']
                elif compare_iso_time(st_iso_time, self.current_time):
                    return SCHEDULE_STATUS['scheduled']
                else: 
                    return SCHEDULE_STATUS['in_progress']
            
        for schedule in self.patient_schedules:
            # Update doctor consultation status
            if schedule.get('waiting_order', -1) < 0:
                schedule['waiting_order'] = -1

            if schedule.get('status') == SCHEDULE_STATUS['cancelled']:
                continue
            
            # To avoid None case of follow-up appointment
            if schedule.get('schedule'):
                tmp_st_iso_time = get_iso_time(schedule['schedule'][0], date=schedule['date'], utc_offset=self._utc_offset)
                tmp_tr_iso_time = get_iso_time(schedule['schedule'][-1], date=schedule['date'], utc_offset=self._utc_offset)
                schedule['status'] = _update_status(tmp_st_iso_time, tmp_tr_iso_time)
            else:
                schedule['status'] = _update_status(is_not_yet=True)

            # Update test schedule status
            for test_info in schedule.get('test') or []:
                if test_info.get('status') == SCHEDULE_STATUS['cancelled']:
                    continue
                
                # To avoid None case
                if test_info.get('schedule'):
                    tmp_st_iso_time = get_iso_time(test_info['schedule'][0], date=test_info['date'], utc_offset=self._utc_offset)
                    tmp_tr_iso_time = get_iso_time(test_info['schedule'][-1], date=test_info['date'], utc_offset=self._utc_offset)
                    test_info['status'] = _update_status(tmp_st_iso_time, tmp_tr_iso_time)
                else:
                    test_info['status'] = _update_status(is_not_yet=True)


    def _reset_variable(self) -> None:
        """
        Reset per-booking variables (currently a no-op).
        """
        pass

    
    def update_env(self, 
                   status: bool,
                   patient_schedule: Union[dict, str],
                   fhir_resources: dict) -> None:
        """
        Record a successful booking: push FHIR resources, store the schedule, advance time, count the doctor's booking.

        Args:
            status (bool): Whether the appointment was successfully assigned; nothing is recorded otherwise.
            patient_schedule (Union[dict, str]): The new schedule (with 'schedule' and 'attending_physician').
            fhir_resources (dict): ``{resource type: resource}`` to create on the FHIR server.
        """
        if status:
            self.update_fhir(fhir_resources)
            self.patient_schedules.append(patient_schedule)
            self.update_current_time()
            
            # To avoid None case of follow-up appointment
            if patient_schedule['schedule']:
                self.booking_num[patient_schedule['attending_physician']] += 1

        self._reset_variable()
 