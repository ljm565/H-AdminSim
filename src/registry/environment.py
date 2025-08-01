import random
from copy import deepcopy
from datetime import datetime
from typing import Union, Tuple
from decimal import Decimal, getcontext

from tasks import FHIRManager
from utils import log
from utils.fhir_utils import convert_fhir_resources_to_doctor_info
from utils.common_utils import (
    get_iso_time,
    get_utc_offset,
    generate_random_iso_time_between,
    convert_time_to_segment
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
        self._tmp_patient_schedules = None
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
        fhir_appointment = [
            x for x in self.fhir_manager.read_all('Appointment', verbose=False)
            if hospital_id in x['resource']['id']
        ]

        # Convert resources regardless of whether they came from cache or fresh read
        doctor_information = convert_fhir_resources_to_doctor_info(
            self._fhir_practitioner_cache,
            self._fhir_practitionerrole_cache,
            self._fhir_schedule_cache,
            self._fhir_slot_cache,
            fhir_appointment
        )
        return doctor_information
    

    def update_fhir(self, fhir_resources: dict):
        """
        Update resources on the FHIR server.

        fhir_resources (dict): Dictionary where each key is a FHIR resource type (e.g., 'Appointment', 'Slot'),
                               and each value is the corresponding FHIR resource data to be updated.
        """
        for resource_type, resource in fhir_resources.items():
            if resource != None and resource_type.lower() in ['patient', 'appointment']:
                self.fhir_manager.create(resource_type, resource, verbose=False)


    def _reschedule_sanity_check(self,
                                 changed_schedule: list[dict],
                                 doctor_information: dict,
                                 patient_condition: dict) -> Tuple[bool, str, dict]:
        """
        Sanity checking codes of LLM rescheduling results. 

        Args:
            changed_schedule (list[dict]): Rescheduling results of existing schedules
            doctor_information (dict): Dictionary of doctor data including their existing schedules.
                                       Each key is a doctor's name, and each value includes a 'schedule' field.
            patient_condition (dict): The conditions of the current patient including duration, priority, etc.

        Returns:
            Tuple[bool, str, dict]:
                - A boolean indicating whether the prediction passed all sanity checks.
                - A string explaining its status.
                - If the sanity check result is `True`, return the `doctor_information` dictionary with the rescheduling results applied; 
                  otherwise, return the original `doctor_information` dictionary.
        """
        if len(changed_schedule) == 0:
            times = dict()
            for schedule in self.patient_schedules:
                e_priority = schedule['priority']
                e_flexibility = schedule['flexibility']
                e_status = schedule['status']
                e_department = schedule['department']
                e_physician = schedule['attending_physician']

                if e_flexibility == 'flexible' and e_status == 'scheduled' and \
                        e_priority > patient_condition['priority'] and e_department == patient_condition['department']:    
                    times.setdefault(e_physician, []).append(schedule['schedule'])

            # Group consecutive time segments for each physician
            for physician, time in times.items():
                consecutive_blocks = list()
                tmp_time_segments = sorted(sum([convert_time_to_segment(self._START_HOUR, self._END_HOUR, self._TIME_UNIT, t) for t in time], []))
                group = [tmp_time_segments[0]]
                for i in range(1, len(tmp_time_segments)):
                    if tmp_time_segments[i] == tmp_time_segments[i - 1] + 1:
                        group.append(tmp_time_segments[i])
                    else:
                        consecutive_blocks.append(group)
                        group = [tmp_time_segments[i]]
                consecutive_blocks.append(group)
                times[physician] = consecutive_blocks
            
            # Check whether rescheduling is possible
            if any(patient_condition['duration'] / self._TIME_UNIT <= len(blocks) for v in times.values() for blocks in v):
                return False, self.status_codes['priority']['priority'], doctor_information

            return True, self.status_codes['correct'], doctor_information     # No changes to check
        
        else:
            sanities, status, e_schedule_idx = list(), list(), list()
            for c_schedule in changed_schedule:
                e_schedule, status_code = None, None
                try:
                    c_schedule_patient = c_schedule['patient']
                    c_duration = float(Decimal(str(c_schedule['schedule'][-1])) - Decimal(str(c_schedule['schedule'][0])))

                    for i, s in enumerate(self.patient_schedules):
                        if s['patient'] == c_schedule_patient:
                            e_schedule_idx.append(i)
                            e_schedule = s
                
                            # Check the format of the changed schedule
                            if set(e_schedule.keys()) != set(c_schedule.keys()):
                                status_code = self.status_codes['format']
                                raise AssertionError
                            
                            # Check the information mathces between the exising and changed schedules
                            for k, v in e_schedule.items():
                                # Rescheduled information should be the same as the existing schedule

                                if k not in ['attending_physician', 'schedule'] and v != c_schedule[k]:
                                    status_code = self.status_codes['information']
                                    raise AssertionError
                                elif k == 'attending_physician':
                                    if e_schedule[k] != c_schedule[k] and e_schedule['department'] != doctor_information[c_schedule[k]]['department']:
                                        status_code = self.status_codes['physician']
                                        raise AssertionError

                            # Check if the priority is higher than the existing schedule (lower is higher priority)
                            if e_schedule['priority'] <= patient_condition['priority']:
                                status_code = self.status_codes['priority']['priority']
                                raise AssertionError

                            # Check if the flexibility is correct
                            if e_schedule['flexibility'] != 'flexible':
                                status_code = self.status_codes['flexibility']
                                raise AssertionError

                            # Check if the status is changable
                            if e_schedule['status'] != 'scheduled':
                                status_code = self.status_codes['status']
                                raise AssertionError

                            # Check if the changed schedule is valid
                            if float(Decimal(str(e_schedule['schedule'][-1])) - Decimal(str(e_schedule['schedule'][0]))) != c_duration:
                                status_code = self.status_codes['schedule']
                                raise AssertionError
                            
                            break
                    
                    # If the patient schedule does not exist, return False
                    if e_schedule is None:
                        status_code = self.status_codes['format']
                        raise AssertionError
                    
                    sanities.append(True)
                    status.append(self.status_codes['correct'])
                
                except KeyError:
                    sanities.append(False)
                    status.append(self.status_codes['format'])
                    continue
                
                except AssertionError:
                    sanities.append(False)
                    status.append(status_code)
                    continue

            # If not all schedules are valid, return False
            for s, c in zip(sanities, status):
                if not s:
                    return False, c, doctor_information
            
            # Check booking priority when rescheduled things are valid  # TODO: idx 이전 스케줄이 변경이 안 되는 경우도 체크 해야하므로 (idx +1)이 아닌 처음부터 봐야할듯
            for idx in e_schedule_idx:
                for _idx in range(idx + 1, len(self.patient_schedules)):
                    # _idx means the booking priority of the patient (higher is lower priority)
                    if _idx not in e_schedule_idx and self.patient_schedules[_idx]['flexibility'] == 'flexible' \
                        and self.patient_schedules[_idx]['status'] == 'scheduled' and self.patient_schedules[_idx]['priority'] > patient_condition['priority']:
                        c_physician = c_schedule['attending_physician']
                        e_department = self.patient_schedules[_idx]['department']
                        c_duration = float(Decimal(str(c_schedule['schedule'][-1])) - Decimal(str(c_schedule['schedule'][0])))
                        e_duration = float(Decimal(str(self.patient_schedules[_idx]['schedule'][-1])) - Decimal(str(self.patient_schedules[_idx]['schedule'][0])))

                        # The case where the lower booking priorty patient has a longer durtion at the same department
                        # If the booking priority is low, handle it conservatively by treating only cases with the same duration as failures.
                        if (e_duration == c_duration and e_department == doctor_information[c_physician]['department']):
                            return False, self.status_codes['priority']['booking'], doctor_information

            # If all the basic reschedules' information are valid, check the time conflicts
            self._tmp_patient_schedules = deepcopy(self.patient_schedules)
            tmp_doctor_information = deepcopy(doctor_information)
            
            ## First, remove the existing schedules that are changed
            for idx in e_schedule_idx:
                original_time = self._tmp_patient_schedules[idx]['schedule']
                original_physician = self._tmp_patient_schedules[idx]['attending_physician']
                tmp_doctor_information[original_physician]['schedule'].remove(original_time)
            
            ## Then, check the time conflicts with the changed schedules
            for idx, c_schedule in zip(e_schedule_idx, changed_schedule):
                changed_time = c_schedule['schedule']
                changed_physician = c_schedule['attending_physician']
                fixed_schedules = tmp_doctor_information[changed_physician]['schedule']

                prediction_schedule_segments = convert_time_to_segment(
                    self._START_HOUR,
                    self._END_HOUR,
                    self._TIME_UNIT,
                    changed_time
                )
                fixed_schedule_segments = sum(
                    [convert_time_to_segment(
                        self._START_HOUR, 
                        self._END_HOUR, 
                        self._TIME_UNIT, 
                        fs
                    ) for fs in fixed_schedules], []
                )
                
                if len(set(prediction_schedule_segments) & set(fixed_schedule_segments)):
                    return False, self.status_codes['conflict'], doctor_information
                
                # Update the temporary patient schedule and doctor information
                self._tmp_patient_schedules[idx] = c_schedule
                tmp_doctor_information[changed_physician]['schedule'].append(changed_time)
                tmp_doctor_information[changed_physician]['schedule'].sort()
            
            return True, self.status_codes['correct'], tmp_doctor_information
        

    def update_current_time(self):
        """
        Update the current hospital time.
        """
        min_iso_time = self.current_time
        max_iso_time = get_iso_time(self.patient_schedules[-1]['schedule'][-1], utc_offset=self._utc_offset)
        self.current_time = generate_random_iso_time_between(min_iso_time, max_iso_time)    # TODO: bug fix when the max_iso_time is smaller than min_iso_time case

    
    def update_patient_status(self):
        """
        Update the status of each patient based on the current hospital time.
        """
        for schedule in self.patient_schedules:
            tmp_st_iso_time = datetime.fromisoformat(get_iso_time(schedule['schedule'][0], utc_offset=self._utc_offset))
            tmp_tr_iso_time = datetime.fromisoformat(get_iso_time(schedule['schedule'][-1], utc_offset=self._utc_offset))
            current_time = datetime.fromisoformat(self.current_time)

            if current_time > tmp_tr_iso_time:
                status = 'completed'
            elif current_time < tmp_st_iso_time:
                status = 'scheduled'
            else: 
                status = 'in_progress'
            
            schedule['status'] = status


    def reset_variable(self):
        """
        Reset variables.
        """
        self._tmp_patient_schedules = None

    
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
            self.patient_schedules = deepcopy(self._tmp_patient_schedules) if self._tmp_patient_schedules != None else self.patient_schedules
            
            if len(self.patient_schedules) and patient_schedule['schedule'][0] > self.patient_schedules[-1]['schedule'][0]:
                self.update_current_time()
            
            self.patient_schedules.append(patient_schedule)
            self.update_patient_status()

        self.reset_variable()
 