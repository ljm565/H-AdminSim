from copy import deepcopy
from decimal import Decimal
from typing import Optional
from langchain.tools import tool
from langchain.agents import AgentExecutor

from .data_converter import DataConverter
from h_adminsim.registry import STATUS_CODES, SCHEDULE_STATUS
from h_adminsim.utils import log
from h_adminsim.utils.fhir_utils import *
from h_adminsim.utils.common_utils import (
    group_consecutive_segments,
    convert_segment_to_time,
    convert_time_to_segment,
    init_result_dict,
    compare_iso_time,
    get_iso_time,
    iso_to_date,
)



class SchedulingRule:
    def __init__(self, 
                 metadata: dict, 
                 department_data: dict, 
                 environment, 
                 fhir_integration: bool = False):
        # Initialize parameters
        self.environment = environment
        self._current_time = self.environment.current_time
        self._utc_offset = self.environment._utc_offset
        self._metadata = metadata
        self._department_data = department_data
        self._START_HOUR = self._metadata.get('time').get('start_hour')
        self._END_HOUR = self._metadata.get('time').get('end_hour')
        self._TIME_UNIT = self._metadata.get('time').get('interval_hour')
        self.current_time = environment.current_time
        self.fhir_integration = fhir_integration


    def physician_filter(self, filtered_doctor_information: dict, preferred_doctor: str) -> list[str]:
        """
        Filter schedules by preferred doctor.

        Args:
            filtered_doctor_information (dict): Filtered doctor information after department filtering.
            preferred_doctor (str): The identifier of the preferred doctor.

        Returns:
            list[str]: A set of candidate schedules that match the preferred doctor.
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

        return list(candidate_schedules)
    

    def date_filter(self, filtered_doctor_information: dict, valid_date: str) -> list[str]:
        """
        Filter schedules by valid date.

        Args:
            filtered_doctor_information (dict): Filtered doctor information after department filtering.
            valid_date (str): The valid date from which to consider schedules.

        Returns:
            list[str]: A set of candidate schedules that are on or after the valid date.
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

        return list(candidate_schedules)
    

    def get_all(self, filtered_doctor_information: dict) -> list[str]:
        """
        Get all candidate schedules without any filtering.

        Args:
            filtered_doctor_information (dict): Filtered doctor information after department filtering.

        Returns:
            list[str]: A set of all candidate schedules without any filtering.
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

        return list(candidate_schedules)
    

    def find_idx(self, 
                 patient_schedule_list: list[dict], 
                 patient_name: str, 
                 doctor_name: str, 
                 status: str,
                 date: Optional[str] = None) -> int:
        """
        Identify the index of the appointment corresponding to the patient's request
        (e.g., cancellation or modification) from the patient's schedule list.

        Args:
            patient_schedule_list (list[dict]): A list of the patient's scheduled appointments.
                                                Each item contains appointment details such as doctor name, date, and time.
            patient_name (str): Name of the patient making the request.
            doctor_name (str): Name of the doctor associated with the target appointment.
            status (str): Patient's appointment status.
            date (Optional[str], optional): Date of the target appointment (YYYY-MM-DD). Defaults to None.

        Returns:
            int: The index of the appointment that matches the patient's request.
        """
        # With FHIR integration case
        if self.fhir_integration:
            # Get patient resource
            name_parts = patient_name.strip().split()
            given = name_parts[0]
            family = name_parts[-1]
            patient_entries = self.environment.fhir_manager.read_all(
                "Patient",
                params={"given": given, "family": family},
                verbose=False,
            )
            if not patient_entries:
                return -1
            patient_id = patient_entries[0]['resource']['id']

            # Get appointment resource
            appointment_entries = self.environment.fhir_manager.read_all(
                'Appointment', 
                params={'patient': f'Patient/{patient_id}'}, 
                verbose=False,
            )
            if not appointment_entries:
                return -1

            candidate_pairs = {
                (entry['resource']['participant'][0]['actor']['display'].lower(),
                 iso_to_date(entry['resource']['start']))
                for entry in appointment_entries
            }

            for idx, patient_schedule in enumerate(patient_schedule_list):
                if patient_schedule['status'] == status and \
                        patient_schedule['patient'].lower() == patient_name.lower() and \
                            patient_schedule['attending_physician'].lower() == doctor_name.lower() and \
                                (date is None or patient_schedule['date'] == date) and \
                                    (patient_schedule['attending_physician'].lower(), patient_schedule['date']) in candidate_pairs:
                    return idx
            return -1

        # Without FHIR integration case
        for idx, patient_schedule in enumerate(patient_schedule_list):
            if patient_schedule['status'] == status and \
                    patient_schedule['patient'].lower() == patient_name.lower() and \
                        patient_schedule['attending_physician'].lower() == doctor_name.lower() and \
                            (date is None or patient_schedule['date'] == date):
                return idx
        return -1


    def find_earliest_time(self, schedules: list[str], delimiter: str = ';;;') -> dict:
        """
        Find the earliest schedule from the list of schedules.

        Args:
            schedules (list[str]): A list of schedules in the format "doctor;;;iso_time".
            delimiter (str, optional): The delimiter used to split doctor and iso_time. Defaults to ';;;'.

        Returns:
            dict: A dictionary containing the earliest doctor(s) and their corresponding schedule(s).
        """
        earliest_doctor, earliest_time = list(), list()

        for schedule in schedules:
            doctor, iso_time = schedule.split(delimiter)

            # skip when the slot is earlier than the current time
            if not compare_iso_time(iso_time, self.current_time):
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
    

    def cancel_schedule(self,
                        idx: int,
                        doctor_info: dict,
                        cancelled_schedule: dict) -> dict:
        """
        Cancel the schedule both in doctor_info and FHIR system.

        Args:
            idx (int): The index of the appointment to be cancelled.
            doctor_info (dict): The doctor information containing schedules.
            cancelled_schedule (dict): The schedule details to be cancelled.

        Returns:
            dict: Updated doctor information after cancellation.
        """
        doctor, date, time = cancelled_schedule['attending_physician'], cancelled_schedule['date'], cancelled_schedule['schedule']
        schedule_list = doctor_info[doctor]['schedule'][date]

        # Remove from doctor_information
        schedule_list.remove(time)  # In-place logic

        # Remove from FHIR
        if self.fhir_integration:
            fhir_appointment = DataConverter.get_fhir_appointment(data={'metadata': deepcopy(self._metadata),
                                                                        'department': deepcopy(self._department_data),
                                                                        'information': deepcopy(cancelled_schedule)})
            self.environment.delete_fhir({'Appointment': fhir_appointment})
        
        # Remove from environment patient_schedules
        self.environment.schedule_cancel_event(idx, True)

        return doctor_info


def create_tools(rule: SchedulingRule,
                 doctor_info: dict,
                 patient_schedule_list: Optional[list[dict]] = None,
                 gt_idx: Optional[int] = None,
                 only_schedule_tool: bool = False,
                 reschedule_pipeline: Optional[callable] = None) -> list[tool]:
    @tool
    def physician_filter_tool(preferred_doctor: str) -> dict:
        """
        Return the earliest available schedule for a preferred doctor.
        
        Args:
            preferred_doctor: Name of the preferred doctor

        Returns:
            dict: The earliest physician-filtered time slot and its information.
        """
        log(f'[TOOL CALL] physician_filter_tool | preferred_doctor={preferred_doctor}', color=True)
        prefix = 'Dr.'
        if prefix not in preferred_doctor:
            preferred_doctor = f'{prefix} {preferred_doctor}'
        schedules = rule.physician_filter(doctor_info, preferred_doctor)
        schedule = rule.find_earliest_time(schedules)
        return schedule

    @tool
    def date_filter_tool(valid_date: str) -> dict:
        """
        Return the earliest available schedule after a specific date.
        
        Args:
            valid_date: Date in YYYY-MM-DD format.
        
        Returns:
            dict: The earliest date-filtered time slot and its information.
        """
        log(f'[TOOL CALL] date_filter_tool | valid_date={valid_date}', color=True)
        schedules = rule.date_filter(doctor_info, valid_date)
        schedule = rule.find_earliest_time(schedules)
        return schedule

    @tool
    def get_all_time_tool() -> dict:
        """
        Return the earliest available schedule among the all available time slots.

        Returns:
            dict: The earliest time slot and its information.
        """
        log(f'[TOOL CALL] get_all_time_tool', color=True)
        schedules = rule.get_all(doctor_info)
        schedule = rule.find_earliest_time(schedules)
        return schedule

    @tool
    def cancel_tool(patient_name: str, doctor_name: str, date: str) -> dict:
        """
        Identify the index of the appointment to be cancelled from the patient's schedule list.

        Args:
            patient_name (str): Name of the patient requesting the cancellation.
            doctor_name (str): Name of the doctor for the appointment to be cancelled.
            date (str): Date of the appointment to be cancelled (YYYY-MM-DD).

        Returns:
            dict: A dictionary containing the cancelled_schedule, result_dict, and updated_doctor_info.
        """
        log(f'[TOOL CALL] cancel_tool | patient_name={patient_name}, doctor_name={doctor_name}, date={date}', color=True)
        updated_doctor_info, cancelled_schedule = None, None
        prefix = 'Dr.'
        if prefix not in doctor_name:
            doctor_name = f'{prefix} {doctor_name}'
        index = rule.find_idx(patient_schedule_list, patient_name, doctor_name, status=SCHEDULE_STATUS['scheduled'], date=date)

        # Update result_dict
        if index == -1 or gt_idx is None:
            status = None
        elif index == gt_idx:
            status = True
        else:
            status = False

        # Update the schedule only when the cancellation is correct or there is no gt_idx
        if index != -1 and (gt_idx is None or status):
            cancelled_schedule = patient_schedule_list[index]
            updated_doctor_info = rule.cancel_schedule(index, doctor_info, cancelled_schedule)
                
        return {'cancelled_schedule': cancelled_schedule, 'updated_doctor_info': updated_doctor_info, 'index': {'gt': gt_idx, 'pred': index}, 'status': status}
    

    @tool
    def reschedule_tool(patient_name: str, doctor_name: str, date: str) -> dict:
        """
        Identify the original appointment to be rescheduled and run the post-retrieval rescheduling pipeline.

        Args:
            patient_name (str): Name of the patient requesting the rescheduling.
            doctor_name (str): Name of the doctor for the appointment to be rescheduled.
            date (str): Date of the original appointment to be rescheduled (YYYY-MM-DD).

        Returns:
            dict: A dictionary containing original_schedule, new_schedule, index, status,
                  pipeline action, and optional schedule_status_code.
        """
        log(f'[TOOL CALL] reschedule_tool | patient_name={patient_name}, doctor_name={doctor_name}, date={date}', color=True)
        prefix = 'Dr.'
        if prefix not in doctor_name:
            doctor_name = f'{prefix} {doctor_name}'
        index = rule.find_idx(patient_schedule_list, patient_name, doctor_name, status=SCHEDULE_STATUS['scheduled'], date=date)

        if index == -1 or gt_idx is None:
            status = None
        elif index == gt_idx:
            status = True
        else:
            status = False

        original_schedule, new_schedule = None, None
        action, schedule_status_code = None, None

        # Run pipeline only when retrieval succeeded (No-GT or correct GT match)
        if index != -1 and (gt_idx is None or status):
            original_schedule = patient_schedule_list[index]
            if reschedule_pipeline is not None:
                pipe_result = reschedule_pipeline(index, original_schedule)
                action = pipe_result['action']
                new_schedule = pipe_result['new_schedule']
                schedule_status_code = pipe_result.get('status_code')

        return {
            'original_schedule': original_schedule,
            'new_schedule': new_schedule,
            'index': {'gt': gt_idx, 'pred': index},
            'status': status,
            'action': action,
            'schedule_status_code': schedule_status_code,
        }
    

    @tool
    def retrieve_patient_tests(patient_name: str, doctor_name: str) -> dict:
        """
        Retrieve the list of tests required for a patient based on their scheduled appointment.

        Args:
            patient_name (str): Name of the patient.
            doctor_name (str): Name of the doctor for the scheduled appointment.

        Returns:
            dict: A dictionary containing the test_list, patient_fv, index, and status.
        """
        log(f'[TOOL CALL] retrieve_patient_tests | patient_name={patient_name}, doctor_name={doctor_name}', color=True)
        test_list, patient_fv = None, None
        prefix = 'Dr.'
        if prefix not in doctor_name:
            doctor_name = f'{prefix} {doctor_name}'
        index = rule.find_idx(patient_schedule_list, patient_name, doctor_name, status=SCHEDULE_STATUS['completed'])

        if index == -1 or gt_idx is None:
            status = None
        elif index == gt_idx:
            status = True
        else:
            status = False

        if index != -1 and (gt_idx is None or status):
            patient_fv = patient_schedule_list[index]
            test_list = patient_schedule_list[index]['test']

        return {
            'test_list': test_list,
            'patient_fv': patient_fv,
            'index': {'gt': gt_idx, 'pred': index},
            'status': status,
        }
    

    if only_schedule_tool:
        return [physician_filter_tool, date_filter_tool, get_all_time_tool]
    return [
        physician_filter_tool, 
        date_filter_tool, 
        get_all_time_tool, 
        cancel_tool, 
        reschedule_tool,
        retrieve_patient_tests,
    ]



def scheduling_tool_calling(client: AgentExecutor, 
                            user_prompt: str,
                            history: list = [],
                            callback = None) -> dict:
    """
    Make an appointment using tool-calling agent.

    Args:
        client (AgentExecutor): The agent executor to handle tool calls.
        user_prompt (str): User prompt used for tool calling.
        history (list, optional): A list of LangChain HumanMessage and AIMessage objects. Defaults to [].
        callback (Optional[TokenUsageCallback], optional): Callback to calculate token usages. Defaults to None.
    
    Returns:
        dict: A dictionary containing the scheduled doctor and their corresponding schedule.
    """
    inputs = {
        "input": user_prompt,
        "chat_history": history,
    }
    if callback is not None:
        response = client.invoke(inputs, config={"callbacks": [callback]})
        token_usage = callback.token_usage
    else:
        response = client.invoke(inputs)
        token_usage = {}

    steps = response.get("intermediate_steps") or []

    if len(steps) > 0:
        tool_output = steps[0][1]
        return {"type": "tool", "result": tool_output, "raw": response, "token": token_usage}

    # No tool call happened
    text = response.get("output") or "Could you tell me one more again?"
    return {"type": "text", "result": text, "raw": response, "token": token_usage}
