import random
from copy import deepcopy
from datetime import datetime
from typing import Union, Tuple
from decimal import Decimal, getcontext

from utils.common_utils import (
    get_iso_time,
    get_utc_offset,
    generate_random_iso_time_between,
    convert_time_to_segment
)



class HospitalEnvironment:
    def __init__(self, agent_test_data):
        self.__init_variable(agent_test_data)
        self.status_codes = {
            'department': 'incorrect department',
            'format': 'incorrect format',
            'information': 'information mismatch',
            'physician': 'incorrect physician',
            'schedule': 'invalid schedule',
            'priority': 'lower priority',
            'flexibility': 'invalid flexibility',
            'status': 'invalid status',
            'conflict': 'schedule conflict',
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


    def _changed_schedule_sanity_check(self,
                                       changed_schedule: list[dict],
                                       doctor_information: dict,
                                       patient_condition: dict) -> bool:
        if len(changed_schedule) == 0:
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
                                    if e_schedule[k] != c_schedule[k] and doctor_information[e_schedule[k]]['department'] != doctor_information[c_schedule[k]]['department']:
                                        status_code = self.status_codes['physician']
                                        raise AssertionError

                            # Check if the priority is higher than the existing schedule (lower is higher priority)
                            if e_schedule['priority'] <= patient_condition['priority']:
                                status_code = self.status_codes['priority']
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
            if not all(sanities):
                return False, f"incorrect reschedule results: {' & '.join(status)}", doctor_information

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

                prediction_schedule_segments = convert_time_to_segment(self._START_HOUR,
                                                            self._END_HOUR,
                                                            self._TIME_UNIT,
                                                            changed_time)
                fixed_schedule_segments = sum([convert_time_to_segment(self._START_HOUR, 
                                                                self._END_HOUR, 
                                                                self._TIME_UNIT, 
                                                                fs) for fs in fixed_schedules], [])
                
                if len(set(prediction_schedule_segments) & set(fixed_schedule_segments)):
                    return False, self.status_codes['conflict'], doctor_information
                
                # Update the temporary patient schedule and doctor information
                self._tmp_patient_schedules[idx] = c_schedule
                tmp_doctor_information[changed_physician]['schedule'].append(changed_time)
                tmp_doctor_information[changed_physician]['schedule'].sort()
            
            return True, self.status_codes['correct'], tmp_doctor_information


    # def update_current_time(self):
    #     """
    #     Set the current time to a random point between the current time and the most recent patient's appointment end time.
    #     """
    #     if self.current_time == None:
    #         self.current_time = get_iso_time(
    #             time_hour=random.uniform(max(0, self._START_HOUR - 6), max(0, self._START_HOUR - self._epsilon)),
    #             utc_offset=self._utc_offset
    #         )
    #     else:
    #         min_iso_time = self.current_time
    #         max_iso_time = get_iso_time(self.patient_schedules[-1]['schedule'][-1], utc_offset=self._utc_offset)
    #         self.current_time = generate_random_iso_time_between(min_iso_time, max_iso_time)    # TODO: bug fix when new department patient started after the current time


    def update_current_time(self):
        """
        Set the current time to a random point between the current time and the most recent patient's appointment end time.
        """
        min_iso_time = self.current_time
        max_iso_time = get_iso_time(self.patient_schedules[-1]['schedule'][-1], utc_offset=self._utc_offset)
        self.current_time = generate_random_iso_time_between(min_iso_time, max_iso_time)    # TODO: bug fix when new department patient started after the current time

    
    def update_patient_status(self):
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
        self._tmp_patient_schedules = None

    
    def update_env(self, status: bool, patient_schedule: Union[dict, str]):
        if status:
            self.patient_schedules = deepcopy(self._tmp_patient_schedules) if self._tmp_patient_schedules != None else self.patient_schedules
            
            if len(self.patient_schedules) and patient_schedule['schedule'][0] > self.patient_schedules[-1]['schedule'][0]:
                self.update_current_time()
            
            self.patient_schedules.append(patient_schedule)
            self.update_patient_status()

        self.reset_variable()
 