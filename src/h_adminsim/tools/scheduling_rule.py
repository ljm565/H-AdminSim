from copy import deepcopy
from decimal import Decimal
from collections import defaultdict
from langchain.tools import tool
from langchain.agents import AgentExecutor
from typing import Union, Optional, TYPE_CHECKING

from .data_converter import DataConverter
from h_adminsim.registry import SCHEDULE_STATUS
from h_adminsim.registry.errors import SchedulingError
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.fhir_utils import *
from h_adminsim.utils.common_utils import (
    group_consecutive_segments,
    convert_segment_to_time,
    convert_time_to_segment,
    compare_iso_time,
    add_hours_to_iso,
    get_iso_time,
    iso_to_hour,
    iso_to_date,
)

if TYPE_CHECKING:
    from h_adminsim.environment.hospital import HospitalEnvironment



class SchedulingRule:
    def __init__(self,
                 metadata: dict,
                 department_data: dict,
                 environment: "HospitalEnvironment",
                 fhir_integration: bool = False):
        
        # Initialize parameters
        self.environment = environment
        self.current_time = self.environment.current_time
        self._utc_offset = self.environment._utc_offset
        self._metadata = metadata
        self._department_data = department_data
        self._START_HOUR = self._metadata.get('time').get('start_hour')
        self._END_HOUR = self._metadata.get('time').get('end_hour')
        self._TIME_UNIT = self._metadata.get('time').get('interval_hour')
        self.fhir_integration = fhir_integration


    def physician_filter(self, 
                         filtered_doctor_information: dict, 
                         preferred_doctor: str, 
                         min_time: Optional[str] = None,
                         max_time: Optional[str] = None) -> list[str]:
        """
        Filter schedules by preferred doctor.

        Args:
            filtered_doctor_information (dict): Filtered doctor information after department filtering.
            preferred_doctor (str): The identifier of the preferred doctor.
            min_time (Optional[str], optional): The earliest acceptable ISO time. Defaults to None.
            max_time (Optional[str], optional): The latest acceptable ISO time. Defaults to None.

        Returns:
            list[str]: A list of candidate schedules that match the preferred doctor.
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
                        if compare_iso_time(free_max_st_iso, self.current_time):
                            candidate_schedules.add(f"{preferred_doctor};;;{free_max_st_iso}")
        
        candidate_schedules = list(candidate_schedules)
        if min_time is not None:
            min_iso_time = get_iso_time(iso_to_hour(min_time), iso_to_date(min_time), self._utc_offset)
            candidate_schedules = [s for s in candidate_schedules if not compare_iso_time(min_iso_time, s.split(';;;')[-1])]
        
        if max_time is not None:
            max_iso_time = get_iso_time(iso_to_hour(max_time), iso_to_date(max_time), self._utc_offset)
            candidate_schedules = [s for s in candidate_schedules if not compare_iso_time(s.split(';;;')[-1], max_iso_time)]

        return candidate_schedules
    

    def date_filter(self, filtered_doctor_information: dict, valid_date: str) -> list[str]:
        """
        Filter schedules by valid date.

        Args:
            filtered_doctor_information (dict): Filtered doctor information after department filtering.
            valid_date (str): The valid date from which to consider schedules.

        Returns:
            list[str]: A list of candidate schedules that are on or after the valid date.
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
                                if compare_iso_time(free_max_st_iso, self.current_time):
                                    candidate_schedules.add(f"{doctor};;;{free_max_st_iso}")

        return list(candidate_schedules)
    

    def get_all(self, filtered_doctor_information: dict) -> list[str]:
        """
        Get all candidate schedules without any filtering.

        Args:
            filtered_doctor_information (dict): Filtered doctor information after department filtering.

        Returns:
            list[str]: A list of all candidate schedules without any filtering.
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
                            if compare_iso_time(free_max_st_iso, self.current_time):
                                candidate_schedules.add(f"{doctor};;;{free_max_st_iso}")

        return list(candidate_schedules)
    

    def find_idx(self, 
                 patient_schedule_list: list[dict], 
                 patient_name: str, 
                 doctor_name: str, 
                 status: Union[str, list[str]],
                 date: Optional[str] = None) -> int:
        """
        Identify the index of the appointment corresponding to the patient's request
        (e.g., cancellation or modification) from the patient's schedule list.

        Args:
            patient_schedule_list (list[dict]): A list of the patient's scheduled appointments.
                                                Each item contains appointment details such as doctor name, date, and time.
            patient_name (str): Name of the patient making the request.
            doctor_name (str): Name of the doctor associated with the target appointment.
            status (Union[str, list[str]]): Patient's appointment status.
            date (Optional[str], optional): Date of the target appointment (YYYY-MM-DD). Defaults to None.

        Returns:
            int: The index of the appointment that matches the patient's request.
        """
        if isinstance(status, str):
            status = [status]

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
                if (patient_schedule['status'] in status and
                    patient_schedule['patient'].lower() == patient_name.lower() and
                    patient_schedule['attending_physician'].lower() == doctor_name.lower() and
                    (date is None or patient_schedule['date'] == date) and
                    (patient_schedule['attending_physician'].lower(), patient_schedule['date']) in candidate_pairs
                ):
                    return idx
            return -1

        # Without FHIR integration case
        for idx, patient_schedule in enumerate(patient_schedule_list):
            if (patient_schedule['status'] in status and
                patient_schedule['patient'].lower() == patient_name.lower() and
                patient_schedule['attending_physician'].lower() == doctor_name.lower() and
                (date is None or patient_schedule['date'] == date)
            ):
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


    def cancel_test_schedule(self,
                             idx: int,
                             doctor_info: dict,
                             test_device_information: dict,
                             cancelled_schedule: dict) -> dict:
        """
        Cancel a whole follow-up booking (all prescribed tests and the follow-up consultation)
        both in the schedule stores and FHIR system.

        Args:
            idx (int): The index of the follow-up booking to be cancelled.
            doctor_info (dict): The doctor information containing schedules.
            test_device_information (dict): Test device information containing device schedules.
            cancelled_schedule (dict): The follow-up booking details to be cancelled.

        Returns:
            dict: Updated doctor information after cancellation.
        """
        # Free every prescribed test's device slot
        device_to_schedule = {device: info['schedule'] for _, tests in test_device_information.items() for test in tests for device, info in test['devices'].items()}
        for entry in cancelled_schedule.get('test') or []:
            device, date, slot = entry['device'], entry['date'], entry['schedule']
            if slot in device_to_schedule.get(device, {}).get(date, []):
                device_to_schedule[device][date].remove(slot)

        # Free the follow-up consultation doctor slot when it was booked in range
        if cancelled_schedule.get('schedule'):
            doctor, date, time = cancelled_schedule['attending_physician'], cancelled_schedule['date'], cancelled_schedule['schedule']
            doctor_info[doctor]['schedule'][date].remove(time)

        # Remove from FHIR (every test appointment, plus the follow-up consultation when present)
        if self.fhir_integration:
            data = {'metadata': deepcopy(self._metadata),
                    'department': deepcopy(self._department_data),
                    'information': deepcopy(cancelled_schedule)}
            fhir_appointments = DataConverter.get_fhir_test_appointments(data)
            if cancelled_schedule.get('schedule'):
                fhir_appointments.append(DataConverter.get_fhir_appointment(data=data))
            for fhir_appointment in fhir_appointments:
                self.environment.delete_fhir({'Appointment': fhir_appointment})

        # Remove from environment patient_schedules
        self.environment.schedule_cancel_event(idx, True)
        # A follow-up booked without an in-range consultation slot never incremented booking_num; keep it consistent.
        if not cancelled_schedule.get('schedule'):
            self.environment.booking_num[cancelled_schedule['attending_physician']] += 1

        return doctor_info


    def _enumerate_device_slots(self,
                                mode: str,
                                test_info: dict,
                                after_iso: str,
                                forbidden_dates: set,
                                extra_busy: Optional[dict] = None,
                                patient_busy: Optional[dict] = None,
                                is_last_test: bool = False,
                                is_last_in_priority_cluster: bool = False,
                                placed_dates: Optional[set] = None) -> list:
        """
        Enumerate candidate (device, date, start_iso, end_iso) windows for a single test.

        Mirrors `physician_filter` slot logic but iterates per device under
        `test_info['devices']` and merges any same-call `extra_busy` bookings
        into the device's pre-existing fixed segments before computing free time.

        Args:
            mode (str): 'throughput_max', 'visit_min', or 'stay_min' — which scheduler this enumeration serves.
                        Affects candidate reduction: `throughput_max`/`visit_min` prune to a few endpoints, while
                        `stay_min` keeps every feasible start so the backtracker can pack tests back-to-back.
            test_info (dict): One entry from `filtered_test_device_information['test'][code]`.
            after_iso (str): Earliest acceptable ISO start time considering priority and depends_on conditions.
            forbidden_dates (set): Dates blocked by an already-placed test in `avoid_same_day`.
            extra_busy (Optional[dict]): `{device_name: {date: [[start_hr, end_hr], ...]}}` of slots booked earlier in this call. Defaults to None.
            patient_busy (Optional[dict]): `{date: [[start_hr, end_hr], ...]}` of intervals during which the patient is already occupied by previously-placed tests. 
                                           Applied to every device (a patient cannot be at two devices at once). Defaults to None.
            is_last_test (bool): Whether this is the last test in the sequence. Defaults to False.
            is_last_in_priority_cluster (bool): True if no subsequent test in `ordered` shares this test's priority.
                                                `throughput_max` additionally collapses to one globally earliest candidate; `visit_min` collapses to one earliest per date; `stay_min` performs no such collapse (all feasible starts are kept). Defaults to False.
            placed_dates (Optional[set]): Set of dates already used by previously-placed tests in this branch.
                                          Only consulted when `is_last_test and mode == 'visit_min'` to keep at most one
                                          candidate (existing-date earliest if any, else new-date earliest). Defaults to None.

        Returns:
            list[tuple[str, str, str, str]]: `(device_name, date, start_iso, end_iso)` tuples
                                              sorted by `(start_iso, device_name)` ascending.
        """
        candidates = []
        duration = test_info['duration_hour']
        min_time_slot_n = max(1, int(Decimal(str(duration)) / Decimal(str(self._TIME_UNIT))))
        all_time_segments = convert_time_to_segment(self._START_HOUR, self._END_HOUR, self._TIME_UNIT)
        patient_busy = patient_busy or {}

        for device_name, device_info in test_info.get('devices', {}).items():
            stop_device_scan = False
            device_schedule = device_info.get('schedule', {})
            extra = (extra_busy or {}).get(device_name, {})

            for date in sorted(set(device_schedule.keys())):
                if date in forbidden_dates:
                    continue

                fixed_intervals = list(device_schedule.get(date, [])) + list(extra.get(date, [])) + list(patient_busy.get(date, []))
                fixed_segments = sum(
                    [convert_time_to_segment(self._START_HOUR, self._END_HOUR, self._TIME_UNIT, fs)
                     for fs in fixed_intervals],
                    [],
                )
                free_time = [s for s in range(len(all_time_segments)) if s not in fixed_segments]
                if not free_time:
                    continue

                runs = [run for run in group_consecutive_segments(free_time) if len(run) >= min_time_slot_n]
                for run in runs:
                    # `stay_min`: keep every feasible start; an interior slot may hug another same-day test
                    if mode == 'stay_min':
                        for i in range(len(run) - min_time_slot_n + 1):
                            window = run[i:i + min_time_slot_n]
                            start_hr, end_hr = convert_segment_to_time(
                                self._START_HOUR, self._END_HOUR, self._TIME_UNIT, window
                            )
                            start_iso = get_iso_time(start_hr, date, utc_offset=self._utc_offset)
                            end_iso = get_iso_time(end_hr, date, utc_offset=self._utc_offset)
                            if compare_iso_time(after_iso, start_iso) or not compare_iso_time(start_iso, self.current_time):
                                continue
                            candidates.append((device_name, date, start_iso, end_iso))
                        continue

                    # Keep both endpoints of each run as candidates: the earliest and the latest slot for the feasibility
                    earliest_idx = None
                    for i in range(len(run) - min_time_slot_n + 1):
                        # Append the earliest available device schedule slot
                        window = run[i:i + min_time_slot_n]
                        start_hr, end_hr = convert_segment_to_time(
                            self._START_HOUR, self._END_HOUR, self._TIME_UNIT, window
                        )
                        start_iso = get_iso_time(start_hr, date, utc_offset=self._utc_offset)
                        end_iso = get_iso_time(end_hr, date, utc_offset=self._utc_offset)
                        if compare_iso_time(after_iso, start_iso) or not compare_iso_time(start_iso, self.current_time):
                            continue
                        candidates.append((device_name, date, start_iso, end_iso))
                        earliest_idx = i
                        break
                    
                    # If no available earliest slot, skip the latest slot check since it also must be not available
                    if earliest_idx is None:
                        continue

                    # Last in its priority cluster: no later test needs the latest slot, so keep only the earliest
                    if is_last_in_priority_cluster:
                        stop_device_scan = True
                        break
                    
                    # Append the latest available device schedule slot
                    latest_idx = len(run) - min_time_slot_n
                    if latest_idx != earliest_idx:
                        window = run[latest_idx:latest_idx + min_time_slot_n]
                        start_hr, end_hr = convert_segment_to_time(
                            self._START_HOUR, self._END_HOUR, self._TIME_UNIT, window
                        )
                        start_iso = get_iso_time(start_hr, date, utc_offset=self._utc_offset)
                        end_iso = get_iso_time(end_hr, date, utc_offset=self._utc_offset)
                        candidates.append((device_name, date, start_iso, end_iso))

                # `throughput_max` only needs the single earliest slot here, so stop scanning further dates.
                if stop_device_scan and mode == 'throughput_max':
                    break
        
        # Sort in order of `start_iso` and `device_name`
        candidates.sort(key=lambda x: (x[2], x[0]))

        if mode == 'throughput_max':
            if is_last_in_priority_cluster:
                candidates = candidates[:1]  # earliest dominates — see `is_last_in_priority_cluster` docstring
        
        elif mode == 'visit_min':
            # For the last test
            if is_last_test:
                placed = placed_dates or set()
                existing_earliest, new_earliest = None, None
                for c in candidates:
                    # If an existing date exists, the earliest of that date (if date in placed)
                    if c[1] in placed:
                        existing_earliest = c
                        break
                    
                    # Otherwise, the earliest in the whole (new_earliest)
                    if new_earliest is None:
                        new_earliest = c
                
                # Pick the existing-date earliest if any, else the global earliest
                candidates = [existing_earliest] if existing_earliest else ([new_earliest] if new_earliest else [])
            
            elif is_last_in_priority_cluster:
                # Remain the earliest candidate per date
                seen_dates = set()
                pruned = []
                for c in candidates:    # already sorted by (start_iso, device)
                    if c[1] not in seen_dates:
                        pruned.append(c)
                        seen_dates.add(c[1])
                candidates = pruned
        
        elif mode == 'stay_min':
            # No reduction: an interior slot may be optimal to hug another same-day test, so keep all
            pass
        
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return candidates


    @staticmethod
    def _topological_order(tests: dict) -> tuple:
        """
        Kahn's topological sort over the `depends_on` DAG restricted to `tests`.

        Args:
            tests (dict): `{test_code: test_info}` to be ordered.

        Returns:
            tuple[list[str], list[str]]: Topologically ordered test codes and any codes
                                         left over due to a cycle.
        """
        def sort_key(c):
            # Stable per-level priority: 
            #   - lower priority int first: Hard constraint
            #   - longer result_hours first: Jackson's rule / Longest Delivery Time (LDT) first
            #   - longer duration_hour first: Longest Processing Time (LPT) first
            info = tests[c]
            return (info.get('priority', 0),
                    -info.get('result_hours', 0),
                    -info.get('duration_hour', 0))

        in_deg = {code: 0 for code in tests}
        children = defaultdict(list)
        for code, info in tests.items():
            for dep in info.get('depends_on', []) or []:
                if dep in tests:
                    in_deg[code] += 1
                    children[dep].append(code)

        ordered = []
        frontier = [c for c, d in in_deg.items() if d == 0]
        frontier.sort(key=sort_key)
        while frontier:
            code = frontier.pop(0)
            ordered.append(code)
            for child in children[code]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    frontier.append(child)
            frontier.sort(key=sort_key)

        unresolved = [c for c, d in in_deg.items() if d > 0]
        return ordered, unresolved


    @staticmethod
    def _build_avoid_pairs(tests: dict) -> dict:
        """
        Symmetric closure of `avoid_same_day` restricted to `tests`.

        Args:
            tests (dict): `{test_code: test_info}`.

        Returns:
            dict: `{test_code: set(test_codes the test must not share a date with)}`.
        """
        avoid = {code: set() for code in tests}
        for code, info in tests.items():
            for other in info.get('avoid_same_day', []) or []:
                if other in tests and other != code:
                    avoid[code].add(other)
                    avoid[other].add(code)
        return avoid


    @staticmethod
    def _check_priority_depends_consistency(tests: dict) -> list:
        """
        Detect `depends_on` edges that contradict the priority ordering.

        Priority is a hard time constraint: `priority(A) < priority(B)` implies
        `A` finishes no later than `B` starts. So a test cannot depend on
        another test that has a strictly higher (numerically larger) priority,
        because that would force the depended-on test to be scheduled later in
        time than the dependent — contradicting both rules at once.

        Args:
            tests (dict): `{test_code: test_info}`.

        Returns:
            list[str]: Codes of tests that violate priority <-> depends_on consistency.
        """
        offenders = []
        for code, info in tests.items():
            p = info.get('priority', 0)
            for dep in info.get('depends_on') or []:
                if dep in tests and tests[dep].get('priority', 0) > p:
                    offenders.append(code)
                    break
        return offenders


    @staticmethod
    def _idle_wait(assignments, priority_floor: float = float('inf')) -> float:
        """
        Idle waiting hours the patient spends between consecutive same-day tests.

        For each visit date the placed tests are sorted by start time and the gap
        between one test's end and the next test's start is summed. Tests on
        different dates contribute nothing (the patient goes home between visits),
        so splitting tests across days is always at least as good as bunching them
        with a gap. With the default `priority_floor` this is exactly the objective
        `stay_min` minimizes.

        `priority_floor` turns this into an admissible lower bound for the
        branch-and-bound search: a gap `(prev, cur)` is only counted when
        `cur['priority'] < priority_floor`. Passing the minimum priority among the
        still-unplaced tests as the floor keeps only the gaps that no remaining test
        can ever fill — any remaining test has priority `>= priority_floor > cur`, so
        the hard ordering `cur.end <= remaining.start` pushes it strictly after `cur`,
        leaving the gap intact in every completion. Same-priority (`== floor`) gaps
        stay excluded because a same-priority test could still slot in and shrink them.

        Args:
            assignments (Iterable[dict]): Placed test assignments, each carrying
                                          `date`, `start`, `end`, and `priority` fields.
            priority_floor (float, optional): Only count gaps whose right test has a
                                              strictly smaller priority. Defaults to +inf
                                              (count every gap → the true objective).

        Returns:
            float: Total (or priority-locked) idle waiting time in hours.
        """
        by_date = defaultdict(list)
        for a in assignments:
            by_date[a['date']].append(a)
        total = 0.0
        for items in by_date.values():
            items.sort(key=lambda a: a['start'])
            for prev, cur in zip(items, items[1:]):
                if cur['priority'] >= priority_floor:
                    continue
                gap = iso_to_hour(cur['start']) - iso_to_hour(prev['end'])
                if gap > 0:
                    total += gap
        return total


    def _backtrack_schedule(self,
                            tests: dict,
                            ordered: list,
                            avoid: dict,
                            mode: str) -> dict:
        """
        Branch-and-bound backtracking search across slot/device choices.

        Tries every (device, date, start) candidate per test in `ordered`; if a
        choice blocks a later test, undoes it and tries the next candidate. Also
        considers a "skip" branch per test so the search degrades gracefully to
        a partial schedule when no full assignment is feasible.

        Objective (lex-min):

        - `mode == 'throughput_max'`: `(per_priority_unscheduled, max_result_ready_at)`.
        - `mode == 'visit_min'`: `(per_priority_unscheduled, num_test_visit_dates, max_result_ready_at)`.
        - `mode == 'stay_min'`: `(per_priority_unscheduled, total_idle_wait_hours, max_result_ready_at)`,
          where `total_idle_wait_hours` is the summed gap between consecutive same-day tests
          (see `_idle_wait`); splitting tests across days incurs no waiting. Its branch-and-bound
          lower bound counts only priority-locked gaps (`_idle_wait` with a `priority_floor`).

        `per_priority_unscheduled` is a tuple indexed by priority ascending
        (lower number first), so the search first protects the most important
        tests from being dropped before optimizing the secondary objectives.

        Priority time ordering is enforced by computing `after_iso` per test
        as `max(current_time, every lower-priority placed test's end, every
        dep's result_ready_at)` — any candidate before that is rejected by
        `_enumerate_device_slots`.

        Args:
            tests (dict): `{test_code: test_info}`.
            ordered (list[str]): Processing order from `_topological_order`.
            avoid (dict): Symmetric closure of `avoid_same_day`.
            mode (str): `'throughput_max'`, `'visit_min'`, or `'stay_min'`.

        Returns:
            dict: `{'placed': {code: assignment}, 'unscheduled': [codes], 'objective': tuple}`.
        """
        assigned = {}
        just_booked = defaultdict(lambda: defaultdict(list))
        best = {'placed': {}, 'unscheduled': list(ordered), 'objective': None}

        # Distinct priorities ascending for the per-priority unscheduled tuple
        distinct_priorities = sorted({tests[c].get('priority', 0) for c in ordered})
        priority_to_idx = {p: i for i, p in enumerate(distinct_priorities)}
        n_priorities = len(distinct_priorities)

        # Because the floor_priority argument looks up the index i+1
        suffix_min_priority = [float('inf')] * (len(ordered) + 1)
        for i in range(len(ordered) - 1, -1, -1):
            suffix_min_priority[i] = min(tests[ordered[i]].get('priority', 0), suffix_min_priority[i + 1])

        def pu_tuple(idx, extra_skip_priority=None):
            """Per-priority unscheduled count for ordered[0..idx-1] given the current
            `assigned` state, optionally adding 1 at `extra_skip_priority`."""
            counts = [0] * n_priorities
            for i in range(idx):
                c = ordered[i]
                if c not in assigned:
                    counts[priority_to_idx[tests[c].get('priority', 0)]] += 1
            if extra_skip_priority is not None:
                counts[priority_to_idx[extra_skip_priority]] += 1
            return tuple(counts)

        def compute_after_iso(code):
            info = tests[code]
            priority = info.get('priority', 0)
            after_iso = self.current_time
            for placed in assigned.values():
                if placed['priority'] < priority and compare_iso_time(placed['end'], after_iso):
                    after_iso = placed['end']
            for dep in info.get('depends_on') or []:
                if dep in assigned and compare_iso_time(assigned[dep]['result_ready_at'], after_iso):
                    after_iso = assigned[dep]['result_ready_at']
            return after_iso

        def leaf_objective():
            pu = pu_tuple(len(ordered))
            if assigned:
                max_ready = max(a['result_ready_at'] for a in assigned.values())
                num_dates = len({a['date'] for a in assigned.values()})
            else:
                max_ready = ''
                num_dates = 0
            if mode == 'throughput_max':
                return (pu, max_ready)
            if mode == 'stay_min':
                return (pu, self._idle_wait(assigned.values()), max_ready)
            return (pu, num_dates, max_ready)

        def _future_max_ready_lb(hypo_end, hypo_ready, hypo_pri, base_max_ready, start_idx):
            """Accumulate per-future-test minimum-possible `result_ready_at` for `ordered[start_idx..]`,
            assuming each is eventually placed at its earliest feasible time given only priority and
            depends_on constraints (resource/avoid_same_day/patient_busy are ignored, all of which only
            push starts later — so the bound stays a true lower bound).

            `hypo_*` dicts seed the hypothetical placement state (existing assigned + possibly the
            current candidate). They are mutated in place as each future test's LB is folded in,
            allowing later tests in the same call to chain off earlier-computed bounds.
            """
            max_lb = base_max_ready
            for j in range(start_idx, len(ordered)):
                u_code = ordered[j]
                if u_code in hypo_ready:
                    continue
                u_info = tests[u_code]
                u_priority = u_info.get('priority', 0)
                u_after = self.current_time
                # Priority hard constraint: u must start after every lower-priority hypothetical end.
                for c, e in hypo_end.items():
                    if hypo_pri[c] < u_priority and compare_iso_time(e, u_after):
                        u_after = e
                # depends_on: u must start after each dep's result_ready_at (real or LB).
                for dep in u_info.get('depends_on') or []:
                    if dep in hypo_ready and compare_iso_time(hypo_ready[dep], u_after):
                        u_after = hypo_ready[dep]
                u_end_lb = add_hours_to_iso(u_after, u_info.get('duration_hour', 0))
                u_lb_ready = add_hours_to_iso(u_end_lb, u_info.get('result_hours', 0))
                hypo_end[u_code] = u_end_lb
                hypo_ready[u_code] = u_lb_ready
                hypo_pri[u_code] = u_priority
                if compare_iso_time(u_lb_ready, max_lb):
                    max_lb = u_lb_ready
            return max_lb

        def lb_with_placement(idx, ready, start_iso, end_iso, date):
            # Place `code` here and fold in the minimum-possible result_ready_at of every remaining test via `_future_max_ready_lb`
            pu = pu_tuple(idx)
            base = max([a['result_ready_at'] for a in assigned.values()] + [ready])

            # Get hypothetical schedule states
            hypo_end = {c: a['end'] for c, a in assigned.items()}
            hypo_ready = {c: a['result_ready_at'] for c, a in assigned.items()}
            hypo_pri = {c: a['priority'] for c, a in assigned.items()}
            code = ordered[idx]
            hypo_end[code] = end_iso
            hypo_ready[code] = ready
            hypo_pri[code] = tests[code].get('priority', 0)

            hyp_ready = _future_max_ready_lb(hypo_end, hypo_ready, hypo_pri, base, idx + 1)

            if mode == 'throughput_max':
                return (pu, hyp_ready)
            if mode == 'stay_min':
                # Lower bound = idle gaps locked in by priority (see `_idle_wait`): among
                # already-placed tests plus this candidate, only gaps a remaining test can never fill.
                cur = {'date': date, 'start': start_iso, 'end': end_iso, 'priority': tests[code].get('priority', 0)}
                locked = self._idle_wait(list(assigned.values()) + [cur], priority_floor=suffix_min_priority[idx + 1])
                return (pu, locked, hyp_ready)
            hyp_dates = {a['date'] for a in assigned.values()}
            hyp_dates.add(date)
            return (pu, len(hyp_dates), hyp_ready)

        def lb_with_skip(idx):
            # Skip `code` (do not seed it into the hypothetical placement) but still fold in the minimum-possible result_ready_at of every remaining test
            cur_priority = tests[ordered[idx]].get('priority', 0)
            pu = pu_tuple(idx, extra_skip_priority=cur_priority)
            base = max([a['result_ready_at'] for a in assigned.values()] + [self.current_time])

            hypo_end = {c: a['end'] for c, a in assigned.items()}
            hypo_ready = {c: a['result_ready_at'] for c, a in assigned.items()}
            hypo_pri = {c: a['priority'] for c, a in assigned.items()}

            cur_max_ready = _future_max_ready_lb(hypo_end, hypo_ready, hypo_pri, base, idx + 1)

            if mode == 'throughput_max':
                return (pu, cur_max_ready)
            if mode == 'stay_min':
                # Same priority-locked idle-gap bound as lb_with_placement, over the already-placed tests only.
                locked = self._idle_wait(assigned.values(), priority_floor=suffix_min_priority[idx + 1])
                return (pu, locked, cur_max_ready)
            cur_dates = {a['date'] for a in assigned.values()}
            return (pu, len(cur_dates), cur_max_ready)

        def recurse(idx):
            if idx == len(ordered):
                obj = leaf_objective()
                if best['objective'] is None or obj < best['objective']:
                    best['placed'] = {c: dict(assigned[c]) for c in assigned}
                    best['unscheduled'] = [c for c in ordered if c not in assigned]
                    best['objective'] = obj
                return

            is_last_test = idx == len(ordered) - 1
            is_last_in_priority_cluster = is_last_test or (
                tests[ordered[idx]].get('priority', 0) < tests[ordered[idx + 1]].get('priority', 0)
            )
            code = ordered[idx]
            info = tests[code]
            deps = info.get('depends_on') or []

            # Cascade-skip when an in-set dep wasn't placed in this branch
            if any(dep in tests and dep not in assigned for dep in deps):
                recurse(idx + 1)
                return

            # To set time constraints
            after_iso = compute_after_iso(code) # To meet priority and depends_on conditions
            forbidden = {assigned[a]['date'] for a in avoid.get(code, set()) if a in assigned}  # To meet avoid_same_day condition

            # To avoid already placed schedules
            patient_busy = defaultdict(list)
            for placed in assigned.values():
                patient_busy[placed['date']].append(
                    [iso_to_hour(placed['start']), iso_to_hour(placed['end'])]
                )

            # Device check
            print(idx, info['duration_hour'], info['result_hours'], after_iso)
            placed_dates_set = {a['date'] for a in assigned.values()} if is_last_test else None
            candidates = self._enumerate_device_slots(
                mode=mode,
                test_info=info,
                after_iso=after_iso,
                forbidden_dates=forbidden,
                extra_busy=just_booked,
                patient_busy=patient_busy, 
                is_last_test=is_last_test,
                is_last_in_priority_cluster=is_last_in_priority_cluster,
                placed_dates=placed_dates_set,
            )

            for device, date, start_iso, end_iso in candidates:
                ready = add_hours_to_iso(end_iso, info.get('result_hours', 0))

                # Branch-and-bound prune: LB folds in the minimum-possible `result_ready_at` of every remaining test (priority + depends_on chain)
                if best['objective'] is not None:
                    lb = lb_with_placement(idx, ready, start_iso, end_iso, date)
                    if lb >= best['objective']:
                        continue

                assigned[code] = {
                    'name': info.get('name'),
                    'code': code,
                    'device': device,
                    'date': date,
                    'start': start_iso,
                    'end': end_iso,
                    'result_ready_at': ready,
                    'priority': info.get('priority', 0),
                }
                just_booked[device][date].append([iso_to_hour(start_iso), iso_to_hour(end_iso)])
                recurse(idx + 1)
                just_booked[device][date].pop()
                del assigned[code]

            # Try skipping this test (partial fallback). Prune only if strictly worse.
            if best['objective'] is None:
                recurse(idx + 1)
            # Best is partial: recurse into the skip branch only if dropping this test could lex-beat best.
            else:
                lb = lb_with_skip(idx)
                if lb < best['objective']:
                    recurse(idx + 1)

        recurse(0)
        return best


    def _assemble_schedule_result(self,
                                  placed: dict,
                                  unscheduled_input: list,
                                  unscheduled_search: list) -> dict:
        """
        Build the public return shape from a backtracker's best snapshot.

        Args:
            placed (dict): `{code: assignment}` from `_backtrack_schedule`.
            unscheduled_input (list): Codes that were missing from `filtered_test_device_information`.
            unscheduled_search (list): Codes the backtracker could not place.

        Returns:
            dict: `{'test_schedule', 'test_visit_dates', 'all_results_ready_at', 'unscheduled', 'status'}`.
        """
        unscheduled = sorted(set(unscheduled_input) | set(unscheduled_search))
        test_visit_dates = sorted({a['date'] for a in placed.values()})
        all_ready = None
        for a in placed.values():
            if all_ready is None or compare_iso_time(a['result_ready_at'], all_ready):
                all_ready = a['result_ready_at']
        status = 'partial' if unscheduled else 'ok'
        return {
            'test_schedule': placed,
            'test_visit_dates': test_visit_dates,
            'all_results_ready_at': all_ready,
            'unscheduled': unscheduled,
            'status': status,
        }


    def schedule_tests(self,
                       mode: str,
                       filtered_test_device_information: dict,
                       test_codes: list) -> dict:
        """
        Unified backtracking test scheduler. Searches every (device, date, start)
        combination via `_backtrack_schedule` and returns the lex-minimal assignment
        for the requested `mode`. Priority is a hard time-ordering constraint
        (`priority(A) < priority(B)` implies `A.end <= B.start`); `depends_on` and
        `avoid_same_day` are enforced during the search.

        The objective per mode (all lexicographically minimized, most-important-tests
        protected from being dropped first):

        - `'throughput_max'`: `(per_priority_unscheduled, max_result_ready_at)` — get every
          result ready as early as possible, regardless of visit count or on-site waiting.
        - `'visit_min'`: `(per_priority_unscheduled, num_test_visit_dates, max_result_ready_at)`
          — pack tests into the fewest distinct visit days.
        - `'stay_min'`: `(per_priority_unscheduled, total_idle_wait_hours, max_result_ready_at)`
          — minimize the patient's idle waiting between consecutive same-day tests, so tests
          are split across days rather than left with long same-day gaps.

        Args:
            mode (str): `'throughput_max'`, `'visit_min'`, or `'stay_min'`.
            filtered_test_device_information (dict): Output of `HospitalEnvironment.get_test_device_schedule`.
            test_codes (list[str]): Codes of the tests the patient must take.

        Returns:
            dict: `{'test_schedule', 'test_visit_dates', 'all_results_ready_at', 'unscheduled', 'status'}`
                  (see `_assemble_schedule_result`); the empty-input fast path returns the same
                  keys with empty values.
        """
        empty_result = {
            'test_schedule': {}, 'test_visit_dates': [], 'all_results_ready_at': None,
            'unscheduled': [], 'status': 'ok',
        }
        if not test_codes:
            return empty_result

        tests_index = filtered_test_device_information.get('test', {})
        tests = {c: tests_index[c] for c in test_codes if c in tests_index}
        missing = [c for c in test_codes if c not in tests_index]

        # Sanity check
        ordered, unresolved = self._topological_order(tests)
        if unresolved or missing:
            raise SchedulingError(colorstr(
                'red',
                f'[SchedulingRule] infeasible test graph — unresolvable depends_on: {unresolved}, '
                f'unknown test codes: {missing}'
            ))

        offenders = self._check_priority_depends_consistency(tests)
        if offenders:
            raise SchedulingError(colorstr(
                'red',
                f'[SchedulingRule] priority/depends_on contradiction on tests: {offenders}'
            ))

        avoid = self._build_avoid_pairs(tests)
        result = self._backtrack_schedule(tests, ordered, avoid, mode=mode)
        return self._assemble_schedule_result(result['placed'], missing, result['unscheduled'])


def create_tools(rule: SchedulingRule,
                 doctor_info: dict,
                 patient_schedule_list: Optional[list[dict]] = None,
                 gt_idx: Optional[int] = None,
                 only_schedule_tool: bool = False,
                 reschedule_pipeline: Optional[callable] = None,
                 test_device_information: Optional[dict] = None,
                 required_test_codes: Optional[list] = None) -> list[tool]:
    @tool
    def physician_filter_tool(preferred_doctor: str, min_time: Optional[str] = None, max_time: Optional[str] = None) -> dict:
        """
        Return the earliest available schedule for a preferred doctor.
        
        Args:
            preferred_doctor (str): Name of the preferred doctor
            min_time (Optional[str], optional): The earliest acceptable ISO time. Defaults to None.
            max_time (Optional[str], optional): The latest acceptable ISO time. Defaults to None.

        Returns:
            dict: The earliest physician-filtered time slot and its information.
        """
        log(f'[TOOL CALL] physician_filter_tool | preferred_doctor={preferred_doctor} | min_time={min_time} | max_time={max_time}', color=True)
        prefix = 'Dr.'
        if prefix not in preferred_doctor:
            preferred_doctor = f'{prefix} {preferred_doctor}'
        schedules = rule.physician_filter(doctor_info, preferred_doctor, min_time, max_time)
        schedule = rule.find_earliest_time(schedules)
        return schedule

    @tool
    def date_filter_tool(valid_date: str) -> dict:
        """
        Return the earliest available schedule after a specific date.
        
        Args:
            valid_date (str): Date in YYYY-MM-DD format.
        
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
    def cancel_tests_tool(patient_name: str, doctor_name: str) -> dict:
        """
        Identify the patient's scheduled follow-up test booking and cancel every test in it.

        Args:
            patient_name (str): Name of the patient requesting the cancellation.
            doctor_name (str): Name of the attending physician who ordered the tests.

        Returns:
            dict: A dictionary containing the cancelled_schedule, updated_doctor_info, index, and status.
        """
        log(f'[TOOL CALL] cancel_tests_tool | patient_name={patient_name}, doctor_name={doctor_name}', color=True)
        updated_doctor_info, cancelled_schedule = None, None
        prefix = 'Dr.'
        if prefix not in doctor_name:
            doctor_name = f'{prefix} {doctor_name}'
        index = rule.find_idx(patient_schedule_list, patient_name, doctor_name, status=[SCHEDULE_STATUS['scheduled'], SCHEDULE_STATUS['not_yet']])

        # Update result_dict
        if index == -1 or gt_idx is None:
            status = None
        elif index == gt_idx:
            status = True
        else:
            status = False

        # Cancel the whole booking only when the identification is correct or there is no gt_idx
        if index != -1 and (gt_idx is None or status):
            cancelled_schedule = patient_schedule_list[index]
            updated_doctor_info = rule.cancel_test_schedule(index, doctor_info, test_device_information, cancelled_schedule)

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
    def reschedule_tests_tool(patient_name: str, doctor_name: str) -> dict:
        """
        Identify the patient's scheduled follow-up test booking and run the post-retrieval rescheduling pipeline to move every test earlier.

        Args:
            patient_name (str): Name of the patient requesting the rescheduling.
            doctor_name (str): Name of the attending physician who ordered the tests.

        Returns:
            dict: A dictionary containing original_schedule, new_schedule, index, status,
                  pipeline action, and optional schedule_status_code.
        """
        log(f'[TOOL CALL] reschedule_tests_tool | patient_name={patient_name}, doctor_name={doctor_name}', color=True)
        prefix = 'Dr.'
        if prefix not in doctor_name:
            doctor_name = f'{prefix} {doctor_name}'
        index = rule.find_idx(patient_schedule_list, patient_name, doctor_name, status=[SCHEDULE_STATUS['scheduled'], SCHEDULE_STATUS['not_yet']])

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
    def retrieve_tests_after_first_visit(patient_name: str, doctor_name: str) -> dict:
        """
        Retrieve the follow-up tests prescribed immediately after a completed first physician visit.

        Use this tool only when scheduling newly prescribed diagnostic tests and
        their corresponding follow-up physician visit after the initial consultation.
        Do not use this tool for routine appointment retrieval, or rescheduling existing test appointments.

        Args:
            patient_name (str): Name of the patient.
            doctor_name (str): Name of the doctor for the scheduled appointment.

        Returns:
            dict: A dictionary containing the test_list, patient_fv, index, and status.
        """
        log(f'[TOOL CALL] retrieve_tests_after_first_visit | patient_name={patient_name}, doctor_name={doctor_name}', color=True)
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
            'action': 'retrieval'
        }
    

    @tool
    def follow_up_throughput_max_test_schedule(attending_physician: str) -> dict:
        """
        Schedule each required test at its earliest available slot to MINIMIZE the
        overall time until all results are ready. Visit-date count is NOT minimized;
        the schedule may span multiple separate visits when that yields the earliest
        completion.
        After scheduling the tests, this tool also schedules a follow-up consultation
        appointment with the specified attending physician.

        Use this tool whenever the patient prioritizes SPEED of test completion
        — e.g. "as soon as possible", "ASAP", "quickly", "earliest", "right away",
        "fastest", "want to finish quickly", or any equivalent urgency expression.
        Explicit acceptance of multiple visits is NOT required to use this tool;
        a plain speed preference is sufficient. This is the default choice when the
        patient expresses urgency and says nothing about visit count.

        IMPORTANT:
            DO NOT call this tool if `attending_physician` is unknown, omitted,
            ambiguous, or not explicitly confirmed by the patient.
            This tool must be called only when the attending physician is explicitly
            identified, since a follow-up consultation appointment will be scheduled
            with that physician after all tests are completed.

        Args:
            attending_physician (str):
                Name of the attending physician for whom the follow-up consultation
                appointment should be scheduled after all required tests are completed.
                This argument is mandatory and must be explicitly provided.

        Returns:
            dict: Mapping of test schedules with fields `test_schedule`, `test_visit_dates`,
                `all_results_ready_at`, `unscheduled`, `fu_schedule`, `status`, and `action`.
        """
        log(f'[TOOL CALL] follow_up_throughput_max_test_schedule | attending_physician={attending_physician}', color=True)
        prefix = 'Dr.'
        if prefix not in attending_physician:
            attending_physician = f'{prefix} {attending_physician}'
        result = rule.schedule_tests(
            'throughput_max', test_device_information, required_test_codes
        )
        fu_appn = rule.physician_filter(doctor_info, attending_physician, result['all_results_ready_at'])
        fu_appn = rule.find_earliest_time(fu_appn)
        result['fu_schedule'] = fu_appn
        result['action'] = 'scheduling'
        return result


    @tool
    def follow_up_visit_min_test_schedule(attending_physician: str) -> dict:
        """
        Group the required tests into the SMALLEST possible number of visit days,
        accepting that some tests may finish later than they could individually.
        After scheduling the tests, this tool also schedules a follow-up consultation
        appointment with the specified attending physician.

        Use this tool ONLY when the patient EXPLICITLY asks to minimize the number
        of hospital visits — e.g. "minimize visits", "fewer days", "on the same
        day", "together", "in one trip", "all at once", "one visit", or any
        equivalent phrasing that targets visit-count rather than speed. Do NOT use
        this tool when the patient only expresses a speed/urgency preference; use
        `follow_up_throughput_max_test_schedule` for that case.

        IMPORTANT:
            DO NOT call this tool if `attending_physician` is unknown, omitted,
            ambiguous, or not explicitly confirmed by the patient.
            This tool must be called only when the attending physician is explicitly
            identified, since a follow-up consultation appointment will be scheduled
            with that physician after all tests are completed.

        Args:
            attending_physician (str):
                Name of the attending physician for whom the follow-up consultation
                appointment should be scheduled after all required tests are completed.
                This argument is mandatory and must be explicitly provided.

        Returns:
            dict: Mapping of test schedules with fields `test_schedule`, `test_visit_dates`,
                `all_results_ready_at`, `unscheduled`, `fu_schedule`, `status`, and `action`.
        """
        log(f'[TOOL CALL] follow_up_visit_min_test_schedule | attending_physician={attending_physician}', color=True)
        prefix = 'Dr.'
        if prefix not in attending_physician:
            attending_physician = f'{prefix} {attending_physician}'
        result = rule.schedule_tests(
            'visit_min', test_device_information, required_test_codes
        )
        fu_appn = rule.physician_filter(doctor_info, attending_physician, result['all_results_ready_at'])
        fu_appn = rule.find_earliest_time(fu_appn)
        result['fu_schedule'] = fu_appn
        result['action'] = 'scheduling'
        return result

    
    @tool
    def follow_up_stay_min_test_schedule(attending_physician: str) -> dict:
        """
        Schedule the required tests so as to MINIMIZE the patient's total idle waiting
        time spent at the hospital between consecutive tests on the same day. Two tests
        that could be taken on one day but with a long gap in between are instead split
        across days when that removes the on-site waiting (a test taken on its own day
        adds no waiting). Neither overall completion speed nor visit-day count is
        minimized. After scheduling the tests, this tool also schedules a follow-up
        consultation appointment with the specified attending physician.

        Use this tool ONLY when the patient EXPLICITLY prioritizes not waiting around at
        the hospital between tests — e.g. "I don't want to wait between tests", "minimize
        waiting time", "no long gaps", "don't want to sit around", or any equivalent
        phrasing targeting on-site idle time. Do NOT use this tool for a plain
        speed/urgency preference (use `follow_up_throughput_max_test_schedule`) or a
        fewer-visits preference (use `follow_up_visit_min_test_schedule`).

        IMPORTANT:
            DO NOT call this tool if `attending_physician` is unknown, omitted,
            ambiguous, or not explicitly confirmed by the patient.
            This tool must be called only when the attending physician is explicitly
            identified, since a follow-up consultation appointment will be scheduled
            with that physician after all tests are completed.

        Args:
            attending_physician (str):
                Name of the attending physician for whom the follow-up consultation
                appointment should be scheduled after all required tests are completed.
                This argument is mandatory and must be explicitly provided.

        Returns:
            dict: Mapping of test schedules with fields `test_schedule`, `test_visit_dates`,
                `all_results_ready_at`, `unscheduled`, `fu_schedule`, `status`, and `action`.
        """
        log(f'[TOOL CALL] follow_up_stay_min_test_schedule | attending_physician={attending_physician}', color=True)
        prefix = 'Dr.'
        if prefix not in attending_physician:
            attending_physician = f'{prefix} {attending_physician}'
        result = rule.schedule_tests(
            'stay_min', test_device_information, required_test_codes
        )
        fu_appn = rule.physician_filter(doctor_info, attending_physician, result['all_results_ready_at'])
        fu_appn = rule.find_earliest_time(fu_appn)
        result['fu_schedule'] = fu_appn
        result['action'] = 'scheduling'
        return result

    # Basic tools
    tools = [
        physician_filter_tool,
        date_filter_tool,
        get_all_time_tool,
        cancel_tool,
        cancel_tests_tool,
        reschedule_tool,
        reschedule_tests_tool,
        retrieve_tests_after_first_visit,
    ]

    # Only scheduling tools are needed
    if only_schedule_tool:
        if test_device_information is not None and required_test_codes is not None:
            return [
                follow_up_throughput_max_test_schedule, 
                follow_up_visit_min_test_schedule,
                follow_up_stay_min_test_schedule,
            ]
        return [physician_filter_tool, date_filter_tool, get_all_time_tool]

    # After determine the patient's tests
    if test_device_information is not None and required_test_codes is not None:
        tools = [
            follow_up_throughput_max_test_schedule, 
            follow_up_visit_min_test_schedule,
            follow_up_stay_min_test_schedule,
        ]
    return tools



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
