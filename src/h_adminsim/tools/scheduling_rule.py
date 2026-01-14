from copy import deepcopy
from decimal import Decimal
from typing import Tuple, Union
from langchain.tools import tool
from langchain.agents import AgentExecutor

from h_adminsim.utils.fhir_utils import *
from h_adminsim.utils.common_utils import (
    group_consecutive_segments,
    convert_segment_to_time,
    convert_time_to_segment,
    compare_iso_time,
    get_iso_time,
    iso_to_date,
    iso_to_hour,
)
from h_adminsim.utils import log



class SchedulingRule:
    def __init__(self, metadata, environment):
        self.environment = environment
        self._current_time = self.environment.current_time
        self._utc_offset = self.environment._utc_offset
        self._START_HOUR = metadata.get('time').get('start_hour')
        self._END_HOUR = metadata.get('time').get('end_hour')
        self._TIME_UNIT = metadata.get('time').get('interval_hour')


    def filter_doctor_schedule(self, doctor_information: dict, department: str, express_detail: bool = False) -> dict:
        """
        Filter doctor information by department.

        Args:
            doctor_information (dict): A dictionary containing information about doctors, 
                                       including their department and schedule details.
            department (str): The department name used to filter the doctors.
            express_detail (bool, optional): Whether to express the timetable in detail. Defaults to False.

        Returns:
            dict: A dictionary containing only the doctors who belong to the specified 
                  department, stored under the 'doctor' key.
        """
        filtered_doctor_information = {'doctor': {}}

        for k, v in doctor_information.items():
            if v['department'] == department:
                tmp_schedule = deepcopy(v)
                del tmp_schedule['capacity_per_hour'], tmp_schedule['capacity'], tmp_schedule['gender'], tmp_schedule['telecom'], tmp_schedule['birthDate']
                tmp_schedule['workload'] = f"{round(self.environment.booking_num[k] / v['capacity'] * 100, 2)}%"
                tmp_schedule['outpatient_duration'] = 1 / v['capacity_per_hour']
                filtered_doctor_information['doctor'][k] = tmp_schedule

        if express_detail:
             for _, info in filtered_doctor_information['doctor'].items():
                info['schedule'] = {
                    date: [{'start': s[0], 'end': s[1]} for s in schedule]
                    for date, schedule in info['schedule'].items()
                }

        return filtered_doctor_information
    

    def physician_filter(self, filtered_doctor_information: dict, preferred_doctor: str) -> set:
        """
        Filter schedules by preferred doctor.

        Args:
            filtered_doctor_information (dict): Filtered doctor information after department filtering.
            preferred_doctor (str): The identifier of the preferred doctor.

        Returns:
            set: A set of candidate schedules that match the preferred doctor.
        """
        candidate_schedules = set()
        schedule_info = filtered_doctor_information['doctor'][preferred_doctor]
        schedule_candidates = schedule_info['schedule']
        min_time_slot_n = int(Decimal(str(schedule_info['outpatient_duration'])) / Decimal(str(self._TIME_UNIT)))
        dates = sorted(list(schedule_candidates.keys()))
        for date in dates:
            schedule = schedule_candidates[date]
            fixed_schedule_segments = sum([convert_time_to_segment(self._START_HOUR, 
                                                                    self._END_HOUR, 
                                                                    self._TIME_UNIT, 
                                                                    fs) for fs in schedule], [])
            all_time_segments = convert_time_to_segment(self._START_HOUR, self._END_HOUR, self._TIME_UNIT)
            free_time = [s for s in range(len(all_time_segments)) if s not in fixed_schedule_segments]
            
            if len(free_time):
                valid_time_segments = [seg for seg in group_consecutive_segments(free_time) if len(seg) >= min_time_slot_n]
                for valid_time in valid_time_segments:
                    for i in range(len(valid_time) - min_time_slot_n + 1):
                        time_slot = valid_time[i:i+min_time_slot_n]
                        free_max_st, _ = convert_segment_to_time(self._START_HOUR, self._END_HOUR, self._TIME_UNIT, [time_slot[0]])
                        free_max_st_iso = get_iso_time(free_max_st, date, utc_offset=self._utc_offset)
                        if compare_iso_time(free_max_st_iso, self._current_time):
                            candidate_schedules.add(f"{preferred_doctor};;;{free_max_st_iso}")

        return candidate_schedules
    

    def date_filter(self, filtered_doctor_information: dict, valid_date: str) -> set:
        """
        Filter schedules by valid date.

        Args:
            filtered_doctor_information (dict): Filtered doctor information after department filtering.
            valid_date (str): The valid date from which to consider schedules.

        Returns:
            set: A set of candidate schedules that are on or after the valid date.
        """
        candidate_schedules = set()
        schedule_infos = filtered_doctor_information['doctor']

        for doctor, schedule_info in schedule_infos.items():
            min_time_slot_n = int(Decimal(str(schedule_info['outpatient_duration'])) / Decimal(str(self._TIME_UNIT)))
            dates = sorted(list(schedule_info['schedule'].keys()))

            for date in dates:
                if not compare_iso_time(valid_date, date):
                    schedule = schedule_info['schedule'][date]
                    fixed_schedule_segments = sum([convert_time_to_segment(self._START_HOUR, 
                                                                            self._END_HOUR, 
                                                                            self._TIME_UNIT, 
                                                                            fs) for fs in schedule], [])
                    all_time_segments = convert_time_to_segment(self._START_HOUR, self._END_HOUR, self._TIME_UNIT)
                    free_time = [s for s in range(len(all_time_segments)) if s not in fixed_schedule_segments]
                    
                    if len(free_time):
                        valid_time_segments = [seg for seg in group_consecutive_segments(free_time) if len(seg) >= min_time_slot_n]
                        for valid_time in valid_time_segments:
                            for i in range(len(valid_time) - min_time_slot_n + 1):
                                time_slot = valid_time[i:i+min_time_slot_n]
                                free_max_st, _ = convert_segment_to_time(self._START_HOUR, self._END_HOUR, self._TIME_UNIT, [time_slot[0]])
                                free_max_st_iso = get_iso_time(free_max_st, date, utc_offset=self._utc_offset)
                                if compare_iso_time(free_max_st_iso, self._current_time):
                                    candidate_schedules.add(f"{doctor};;;{free_max_st_iso}")

        return candidate_schedules
    

    def no_filter(self, filtered_doctor_information: dict) -> set:
        """
        Get all candidate schedules without any filtering.

        Args:
            filtered_doctor_information (dict): Filtered doctor information after department filtering.

        Returns:
            set: A set of all candidate schedules without any filtering.
        """
        candidate_schedules = set()
        schedule_infos = filtered_doctor_information['doctor']

        for doctor, schedule_info in schedule_infos.items():
            min_time_slot_n = int(Decimal(str(schedule_info['outpatient_duration'])) / Decimal(str(self._TIME_UNIT)))
            dates = sorted(list(schedule_info['schedule'].keys()))

            for date in dates:
                schedule = schedule_info['schedule'][date]
                fixed_schedule_segments = sum([convert_time_to_segment(self._START_HOUR, 
                                                                        self._END_HOUR, 
                                                                        self._TIME_UNIT, 
                                                                        fs) for fs in schedule], [])
                all_time_segments = convert_time_to_segment(self._START_HOUR, self._END_HOUR, self._TIME_UNIT)
                free_time = [s for s in range(len(all_time_segments)) if s not in fixed_schedule_segments]
                
                if len(free_time):
                    valid_time_segments = [seg for seg in group_consecutive_segments(free_time) if len(seg) >= min_time_slot_n]
                    for valid_time in valid_time_segments:
                        for i in range(len(valid_time) - min_time_slot_n + 1):
                            time_slot = valid_time[i:i+min_time_slot_n]
                            free_max_st, _ = convert_segment_to_time(self._START_HOUR, self._END_HOUR, self._TIME_UNIT, [time_slot[0]])
                            free_max_st_iso = get_iso_time(free_max_st, date, utc_offset=self._utc_offset)
                            if compare_iso_time(free_max_st_iso, self._current_time):
                                candidate_schedules.add(f"{doctor};;;{free_max_st_iso}")

        return candidate_schedules
    

    def find_earliest_time(self, schedules: list[str], current_time: str, delimiter: str = ';;;') -> dict:
        """
        Find the earliest schedule from the list of schedules.

        Args:
            schedules (list[str]): A list of schedules in the format "doctor;;;iso_time".
            current_time (str): Current time of the simulation environment.
            delimiter (str, optional): The delimiter used to split doctor and iso_time. Defaults to ';;;'.

        Returns:
            dict: A dictionary containing the earliest doctor(s) and their corresponding schedule(s).
        """
        earliest_doctor, earliest_time = list(), list()

        for schedule in schedules:
            doctor, iso_time = schedule.split(delimiter)

            # skip when the slot is earlier than the current time
            if not compare_iso_time(iso_time, current_time):
                continue

            if not len(earliest_doctor):
                earliest_doctor.append(doctor) 
                earliest_time.append(iso_time)
                continue
            
            # Append if the iso_time is same with the alrealdy appended one
            if earliest_time[0] == iso_time:
                earliest_doctor.append(doctor)
                earliest_time.append(iso_time)
            
            elif compare_iso_time(earliest_time[0], iso_time):
                earliest_doctor = [doctor]
                earliest_time = [iso_time]
        
        return {'doctor': earliest_doctor, 'schedule': earliest_time}



    def scheduling_rule(self,
                        patient_condition: dict, 
                        doctor_information: dict, 
                        current_time: str,
                        reschedule_flag: bool = False, 
                        verbose: bool = False) -> Tuple[bool, str, Union[str, dict], dict, list[str]]:
        """
        Make an appointment between the doctor and the patient.

        Args:
            patient_condition (dict): The conditions including name, preference, etc.
            doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                       including availability and other relevant details.
            current_time (str): Current time of the simulation environment.        
            reschedule_flag (bool, optional): Whether this process is rescheduling or not. Defaults to False.
            verbose (bool, optional): Whether logging the each result or not. Defaults to False.

        Return
            Tuple[bool, str, Union[str, dict], dict, list[str]]: 
                - A boolean indicating whether the prediction passed all sanity checks.
                - A string explaining its status.
                - The original prediction (wrong case) or processed prediction (correct case) (either unchanged or used for debugging/logging if invalid).
                - Updated doctor information after processing the prediction.
                - A trial information.
        """
        schedule1, schedule2 = set(), set()
        department = patient_condition['department']
        preference = patient_condition['preference'] if isinstance(patient_condition['preference'], list) else [patient_condition['preference']]
        preferred_doctor = patient_condition['preferred_doctor']
        valid_date = patient_condition['valid_from']
        filtered_doctor_information = self.filter_doctor_schedule(doctor_information, department)
        
        basic_schedule = self.no_filter(filtered_doctor_information)
        if 'doctor' in preference:
            schedule1 = self.physician_filter(filtered_doctor_information, preferred_doctor)
        if 'date' in preference:
            schedule2 = self.date_filter(filtered_doctor_information, valid_date)

        if not (len(schedule1) or len(schedule2)):
            schedule = basic_schedule
        else:
            if len(schedule1) and len(schedule2):
                schedule = list(schedule1.intersection(schedule2))
            elif len(schedule1):
                schedule = list(schedule1)
            elif len(schedule2):
                schedule = list(schedule2)
            
        schedule = self.find_earliest_time(schedule, current_time)
        
        doctor = schedule['doctor'][0]
        duration = filtered_doctor_information['doctor'][doctor]['outpatient_duration']
        date, st_hour = iso_to_date(schedule['schedule'][0]), iso_to_hour(schedule['schedule'][0])
        tr_hour = float(Decimal(str(duration)) + Decimal(str(st_hour)))

        return {'schedule': {doctor: {'date': date, 'start': st_hour, 'end': tr_hour}}}



def create_tools(rule: SchedulingRule, filtered_doctor_info: dict) -> list[tool]:
    @tool
    def physician_filter_tool(preferred_doctor: str) -> list:
        """Return available schedules for preferred doctor.
        
        Args:
            preferred_doctor: Name of the preferred doctor
        """
        log(f'[TOOL CALL] physician_filter_tool | preferred_doctor={preferred_doctor}', color=True)
        prefix = 'Dr.'
        if prefix not in preferred_doctor:
            preferred_doctor = f'{prefix} {preferred_doctor}'
        result = rule.physician_filter(filtered_doctor_info, preferred_doctor)
        return list(result)

    @tool
    def date_filter_tool(valid_date: str) -> list:
        """Return schedules for a preferred date.
        
        Args:
            valid_date: Date in YYYY-MM-DD format
        """
        log(f'[TOOL CALL] date_filter_tool | valid_date={valid_date}', color=True)
        result = rule.date_filter(filtered_doctor_info, valid_date)
        return list(result)

    @tool
    def no_filter_tool() -> list:
        """Return all available schedules without filtering."""
        log(f'[TOOL CALL] no_filter_tool', color=True)
        result = rule.no_filter(filtered_doctor_info)
        return list(result)

    return [physician_filter_tool, date_filter_tool, no_filter_tool]



def scheduling_tool_calling(client: AgentExecutor, 
                            rule: SchedulingRule, 
                            known_condition: dict, 
                            doctor_information: dict,
                            current_time: str) -> dict:
    """
    Make an appointment using tool-calling agent.

    Args:
        client (AgentExecutor): The agent executor to handle tool calls.
        rule (SchedulingRule): The scheduling rule instance.
        known_condition (dict): Patient conditions known to the staff.
        doctor_information (dict): A dictionary containing information about the doctor(s) involved,
                                   including availability and other relevant details.
        current_time (str): Current time of the simulation environment.
    
    Returns:
        dict: A dictionary containing the scheduled doctor and their corresponding schedule.
    """
    department = known_condition['department']
    preference = known_condition['patient_intention']

    # Tool calling for candidate slots and find the earliest one
    schedule = client.invoke({
        'input': preference
    })['intermediate_steps'][0][1]
    schedule = rule.find_earliest_time(schedule, current_time)
    
    # Post-processing
    filtered_doctor_information = rule.filter_doctor_schedule(doctor_information, department)
    doctor = schedule['doctor'][0]
    duration = filtered_doctor_information['doctor'][doctor]['outpatient_duration']
    date, st_hour = iso_to_date(schedule['schedule'][0]), iso_to_hour(schedule['schedule'][0])
    tr_hour = float(Decimal(str(duration)) + Decimal(str(st_hour)))
    
    return {'schedule': {doctor: {'date': date, 'start': st_hour, 'end': tr_hour}}}
