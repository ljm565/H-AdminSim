import math
import random
from tqdm import tqdm
from copy import deepcopy
from datetime import timedelta
from importlib import resources
from typing import Tuple, Optional
from collections import defaultdict
from decimal import Decimal, getcontext

from h_adminsim.task.schedule_assign import ScheduleAssigner
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.common_utils import *
from h_adminsim.utils.filesys_utils import *
from h_adminsim.utils.random_utils import (
    generate_random_code,
    generate_random_date,
    generate_random_prob,
    generate_random_telecom,
    generate_random_address,
    generate_random_id_number,
    generate_random_code_with_prob,    
)
from h_adminsim.tools.synthdat.data_synthesizer import DataSynthesizer



class FollowUpDataSynthesizer(DataSynthesizer):
    def __init__(self, config, source_data_dir: Optional[str] = None):
        super().__init__(config, source_data_dir)
        getcontext().prec = 10

        if source_data_dir:
            self.data_save_dir = source_data_dir
            self.source_data_files = sorted(get_files(source_data_dir, ext='json'))
            if not self.source_data_files:
                raise FileNotFoundError(
                    log(f"No hospital JSON files found in {source_data_dir}", "error")
                )
        else:
            self.source_data_files = []


    def synthesize(self, sanity_check: bool = False) -> list[dict]:
        """
        Synthesize follow-up patients.

        If source_data_dir was provided, reads existing hospital_*.json files and
        merges follow-up patients into them. Otherwise, generates hospital infrastructure
        from scratch using DataSynthesizer.define_hospital_info() and adds follow-up
        patients with doctor capacity-based count.

        Args:
            sanity_check (bool, optional): Whether to validate generated data. Defaults to False.

        Returns:
            list[dict]: List of hospital data dicts containing follow_up_visit patients.
        """
        try:
            all_data = []

            if self.source_data_files:
                # Merge mode: add follow-up patients to existing hospital data
                for data_file in tqdm(self.source_data_files, desc='Synthesizing follow-up patient data..'):
                    hospital_data = json_load(data_file)
                    
                    # Make fixed schedule of tests
                    hospital_data['test'] = FollowUpDataSynthesizer.generate_test_schedule(self.config, hospital_data)
                    merged_data = FollowUpDataSynthesizer.generate_followup_patients(self.config, hospital_data)

                    if sanity_check:
                        FollowUpDataSynthesizer._sanity_check(merged_data)

                    json_save_fast(data_file, merged_data)
                    all_data.append(merged_data)

                log(f"Total {len(all_data)} follow-up data merged into existing files. Path: `{self.data_save_dir}`", color=True)

            else:
                # Standalone mode: generate hospital infrastructure + follow-up patients
                hospitals = DataSynthesizer.hospital_list_generator(self.config.hospital_data.hospital_n)
                for i, hospital in tqdm(
                    enumerate(hospitals), 
                    desc='Synthesizing follow-up patient data (standalone)..', 
                    total=len(hospitals)
                ):
                    data = DataSynthesizer.define_hospital_info(self.config, hospital)
                    hospital_data = to_dict(data)
                    
                    # Make fixed schedule of tests
                    hospital_data['test'] = FollowUpDataSynthesizer.generate_test_schedule(self.config, hospital_data)
                    merged_data = FollowUpDataSynthesizer.generate_followup_patients(self.config, hospital_data)

                    if sanity_check:
                        FollowUpDataSynthesizer._sanity_check(merged_data)

                    json_save_fast(
                        self.data_save_dir / f'hospital_{padded_int(i, len(str(self._n)))}.json',
                        merged_data
                    )
                    all_data.append(merged_data)

                log(f"Total {len(all_data)} follow-up data synthesized. Path: `{self.data_save_dir}`", color=True)

            return all_data

        except Exception as e:
            log(f"Follow-up data synthesizing failed: {e}", level='error')
            raise
    

    @staticmethod
    def generate_test_schedule(config, hospital_data: dict) -> dict:
        """
        Generate hospital test schedules and merge them into hospital data.

        Reads hospital metadata and departments information from existing hospital data, 
        generates test schedules and merges them into the same patient dict.

        Args:
            config: Configuration object with hospital_data.follow_up_visit settings.
            hospital_data (dict): Existing hospital data containing metadata and department dicts.

        Returns:
            dict: Hospital data with test schedules merged into the hospital data dict.
        """
        # Hospital time parameters
        metadata = hospital_data['metadata']
        start_hour = float(metadata['time']['start_hour'])
        end_hour = float(metadata['time']['end_hour'])
        interval_hour = float(metadata['time']['interval_hour'])
        start_date = metadata['start_date']
        days = metadata['days']
        hospital_time_segments = convert_time_to_segment(start_hour, end_hour, interval_hour)
        fu_config = config.hospital_data.follow_up_visit
        
        # Initialize eligible tests
        departments = list(hospital_data['department'].keys())
        eligible_tests = {
            d: FollowUpDataSynthesizer.test_list_generator(
                d,
                fu_config.test_per_department.min,
                fu_config.test_per_department.max,
            ) for d in departments
        }

        # Assign random fixed schedules for the tests
        dates = generate_date_range(start_date, days)
        scheduler = ScheduleAssigner(start_hour, end_hour, interval_hour)
        for test_list in eligible_tests.values():
            for _test in test_list:
                device_n = random.randint(fu_config.n_machines_per_test.min, fu_config.n_machines_per_test.max)
                test_duration_segments = max(1, math.ceil(Decimal(str(_test['duration_hour'])) / Decimal(str(interval_hour))))
                _test['device_n'] = device_n
                _test['devices'] = dict()
                for i in range(device_n):
                    device_name = f"{_test['code']}-{i}"
                    _test['devices'][device_name] = dict()
                    test_schedule = {
                        date: scheduler(
                            generate_random_prob(
                                fu_config.test_has_schedule_prob,
                                fu_config.test_fixed_schedule_ratio.min,
                                fu_config.test_fixed_schedule_ratio.max,
                            ),
                            True,
                            hospital_time_segments,
                            min_chunk_size=test_duration_segments,
                            max_chunk_size=test_duration_segments,
                        )[1] for date in dates
                    }
                    _test['devices'][device_name]['schedule'] = test_schedule
        
        return eligible_tests


    @staticmethod
    def generate_followup_patients(config, hospital_data: dict, max_consecutive_failures: int = 3) -> dict:
        """
        Generate follow-up patient profiles and merge them into hospital data.

        Reads hospital metadata, departments, and doctor information from existing hospital data, 
        generates follow-up patients who need medical tests scheduled, and merges them into the same patient dict.

        Args:
            config: Configuration object with hospital_data.follow_up_visit settings.
            hospital_data (dict): Existing hospital data containing
                                   metadata, department, doctor, and patient dicts.
            max_consecutive_failures (int): Maximum number of consecutive failed attempts to generate
                                            a valid test combination before stopping the loop. Defaults to 3.
                               

        Returns:
            dict: Hospital data with follow-up patients merged into the patient dict.
        """
        def _get_available_tests(vacant_test_schedule: dict, func: callable) -> list[Tuple]:
            """
            Get available test keys that satisfy the condition defined by func, sorted by priority and shuffled within the same priority level.

            Args:
                vacant_test_schedule (dict): Dictionary containing vacant test schedules with their details.
                func (callable): A function that takes a test key and its corresponding schedule details, 
                                 and returns True if the test is considered available based on custom conditions.

            Returns:
                list[Tuple]: A list of test keys (priority, test_code, department, device_name) that satisfy the condition defined by func,
                             sorted by priority and shuffled within the same priority level.
            """
            available_keys = sorted(
                [k for k, v in vacant_test_schedule.items() if func(k, v)],
                key=lambda k: k[0],
            )
            grouped = defaultdict(list)
            for k in available_keys:
                grouped[k[0]].append(k)

            shuffled_candidates = []
            for priority in sorted(grouped.keys()):
                group = grouped[priority]
                random.shuffle(group)
                shuffled_candidates.extend(group)
            
            return shuffled_candidates
        

        def _get_available_slots(schedules: list[list[str]], func: callable) -> list[tuple[int, list[str]]]:
            """
            Filter time slots from a schedule list based on a condition function.

            Args:
                schedules (list[list[str]]): List of [start_iso, end_iso] time pairs representing candidate slots.
                func (callable): Filter function that receives (start_iso, end_iso) and returns True
                                 if the slot satisfies the scheduling conditions (e.g. after prev_end_iso,
                                 not on a blocked date, after all depends_on result times).

            Returns:
                list[tuple[int, list[str]]]: List of (original_index, [start_iso, end_iso]) for slots
                                             that passed the filter. The index is needed to remove the
                                             chosen slot from the schedule upon commitment.
            """
            valid_slots = [
                (idx, [st, tr]) for idx, (st, tr) in enumerate(schedules)
                if func(st, tr)
            ]
            return valid_slots
        

        # Hospital time parameters and data
        metadata = hospital_data['metadata']
        start_hour = float(metadata['time']['start_hour'])
        end_hour = float(metadata['time']['end_hour'])
        interval_hour = float(metadata['time']['interval_hour'])
        fixed_test_schedule = hospital_data['test']
        hospital_time_segments = convert_time_to_segment(start_hour, end_hour, interval_hour)

        # Extract hospital data
        doctor_info = hospital_data['doctor']
        patient_info = dict(hospital_data.get('patient', {}))
        department_info = hospital_data['department']

        # Follow-up config
        fu_config = config.hospital_data.follow_up_visit
        tests_min = max(1, fu_config.tests_per_patient.min)
        tests_max = min(fu_config.test_per_department.max, fu_config.tests_per_patient.max)
        cross_dept_prob = fu_config.cross_department_test_prob
        include_consultation = fu_config.get('include_consultation', True)

        # Collect vacant test schedules
        vacant_test_schedules = {}
        for dept, test_list in fixed_test_schedule.items():
            for single_test in test_list:
                for device_name, device_info in single_test['devices'].items():
                    key = (single_test['priority'], single_test['code'], dept ,device_name)
                    test_duration_segments = max(1, math.ceil(Decimal(str(single_test['duration_hour'])) / Decimal(str(interval_hour))))
                    # Initialize
                    if key not in vacant_test_schedules:
                        vacant_test_schedules[key] = {
                            'result_hours': single_test.get('result_hours', 0),
                            'depends_on': single_test.get('depends_on', []),
                            'avoid_same_day': single_test.get('avoid_same_day', []),
                            'schedule': [],
                            'remain_n': 0,
                        }
                    for date, schedules in device_info['schedule'].items():
                        schedule_segments_flat = list()
                        for time_range in schedules:
                            schedule_segments_flat.extend(
                                convert_time_to_segment(start_hour, end_hour, interval_hour, time_range=time_range)
                            )
                        
                        # Compute available patient segments
                        vacant_segments = list(set(hospital_time_segments) - set(schedule_segments_flat))
                        vacant_schedules = [list(convert_segment_to_time(start_hour, end_hour, interval_hour, chunk)) 
                            for chunk in split_into_contiguous_chunks(vacant_segments, test_duration_segments)
                        ]
                        ratio = generate_random_prob(
                            1,
                            fu_config.test_schedule_appointment_ratio.min,
                            fu_config.test_schedule_appointment_ratio.max,
                        )
                        vacant_test_schedules[key]['schedule'].extend([
                            [get_iso_time(vs[0], date), get_iso_time(vs[1], date)] 
                            for vs in vacant_schedules
                        ])
                        vacant_test_schedules[key]['remain_n'] += int(len(vacant_schedules) * ratio)
                    
        # Make all possible test combinations based on the vacant_test_schedule
        all_combinations = []
        consecutive_failures = 0

        while consecutive_failures < max_consecutive_failures:
            # For the first test, only consider those without dependencies
            shuffled_candidates = _get_available_tests(
                vacant_test_schedules, 
                lambda k, v: v['remain_n'] > 0 and len(v['depends_on']) == 0
            )
            if len(shuffled_candidates) < tests_min:
                break

            n_tests = random.randint(tests_min, min(tests_max, len(shuffled_candidates)))

            # Choose a primary department for the patient and allow cross-department tests with a certain probability
            available_depts = sorted(list(set(k[2] for k in shuffled_candidates)))
            primary_dept = random.choice(available_depts)

            # Build one combination in ascending priority order
            combination = []
            prev_end_iso = ''
            used_codes = set()
            impossible_date = defaultdict(list)
            result_out_time = dict()

            for step in range(n_tests):
                # First test: primary dept only; subsequent: cross-dept with probability
                allow_cross = (step > 0) and (random.random() < cross_dept_prob)

                for key in shuffled_candidates:
                    priority, test_code, dept, device_name = key
                    
                    # Cross-dept and priority conditions
                    if (not allow_cross and dept != primary_dept) or \
                        (len(combination) and combination[-1]['priority'] > priority):
                        continue

                    # Filter the valid time slots
                    entry = vacant_test_schedules[key]
                    result_hours, depends_on, avoid_same_day = entry['result_hours'], entry['depends_on'], entry['avoid_same_day']

                    valid_slots = _get_available_slots(
                        entry['schedule'],
                        lambda st, tr, _dep=depends_on: (st > prev_end_iso) and \
                            iso_to_date(st) not in impossible_date.get(test_code, []) and \
                            all(st > result_out_time.get(c, '') for c in _dep)
                    )
                    if not valid_slots:
                        continue

                    chosen_idx, chosen_slot = random.choice(valid_slots)
                    combination.append({
                        'priority': priority,
                        'test_code': test_code,
                        'department': dept,
                        'device_name': device_name,
                        'schedule': chosen_slot,
                        'result_hours': result_hours,
                        'depends_on': depends_on,
                        'avoid_same_day': avoid_same_day,
                        '_key': key,
                        '_slot_idx': chosen_idx,
                    })

                    # Update conditions
                    prev_end_iso = chosen_slot[1]
                    result_out_time[test_code] = (str_to_datetime(prev_end_iso) + timedelta(hours=result_hours)).isoformat()
                    prev_end_date = iso_to_date(prev_end_iso)
                    used_codes.add(test_code)
                    for code in avoid_same_day:
                        impossible_date[code].append(prev_end_date)
                    break
                
                else:
                    # Executed only if no break occurs
                    break  # No valid test found for this step
                
                # Update available candidates depend on previous tests
                shuffled_candidates = _get_available_tests(
                    vacant_test_schedules,
                    lambda k, v: (not k[1] in used_codes) and \
                        (v['remain_n'] > 0) and len(set(v['depends_on']) - used_codes) == 0
                )

            # Validate combination size
            if len(combination) < tests_min:
                consecutive_failures += 1
                continue

            # Commit: remove used slots and decrement remain_n
            for item in combination:
                key = item.pop('_key')
                slot_idx = item.pop('_slot_idx')
                item['date'] = iso_to_date(item['schedule'][0])
                item['schedule'] = [iso_to_hour(item['schedule'][0]), iso_to_hour(item['schedule'][1])]
                vacant_test_schedules[key]['schedule'].pop(slot_idx)
                vacant_test_schedules[key]['remain_n'] -= 1
                if vacant_test_schedules[key]['remain_n'] <= 0:
                    del vacant_test_schedules[key]

            all_combinations.append(combination)
            consecutive_failures = 0

        # Make patient profiles
        follow_up_patient_info = dict()
        used_names = list(set(patient_info.keys()))
        new_names = DataSynthesizer.name_list_generator(len(all_combinations), reject_list=list(used_names))

        # Make sure not to duplicate the first-visit appointments
        _doctor_info = deepcopy(doctor_info)
        for info in patient_info.values():
            _doctor, _date, _schedule = info['attending_physician'], info['date'], info['schedule']
            _doctor_info[_doctor]['schedule'][_date].append(_schedule)

        for name, combination in zip(new_names, all_combinations):
            department = combination[0]['department']
            doctor = random.choice(department_info[department]['doctor'])
            duration = int(Decimal(str(1)) / Decimal(str(doctor_info[doctor]['capacity_per_hour'])) / Decimal(str(interval_hour)))
            symptom_level = generate_random_code_with_prob(
                fu_config.symptom.type,
                fu_config.symptom.probs
            )
            birth_date = generate_random_date()

            if include_consultation:
                last_time = max(
                    [
                        (str_to_datetime(get_iso_time(comb['schedule'][1], comb['date'])) + timedelta(hours=comb['result_hours'])).isoformat() 
                        for comb in combination
                    ]
                )
                last_date, last_schedule = iso_to_date(last_time), iso_to_hour(last_time)

                # Find valid consultation slots for the attending doctor after the last test
                valid_slots = []
                doctor_schedule = _doctor_info[doctor]['schedule']

                for _date in sorted(doctor_schedule.keys()):
                    if _date < last_date:
                        continue

                    occupied_segs = set()
                    for _schedule in doctor_schedule[_date]:
                        occupied_segs.update(
                            convert_time_to_segment(start_hour, end_hour, interval_hour, time_range=_schedule)
                        )

                    available_segs = sorted(set(hospital_time_segments) - occupied_segs)

                    # On last_date, only allow segments at or after last_schedule
                    if _date == last_date:
                        if last_schedule < end_hour:
                            after_segs = set(convert_time_to_segment(
                                start_hour, end_hour, interval_hour, time_range=[max(last_schedule, start_hour), end_hour]
                            ))
                            available_segs = sorted(set(available_segs) & after_segs)
                        else:
                            available_segs = []

                    slot = FollowUpDataSynthesizer._find_contiguous_slot(available_segs, duration)
                    if slot is not None:
                        slot_time = list(convert_segment_to_time(start_hour, end_hour, interval_hour, slot))
                        valid_slots.append((_date, slot_time))

                if valid_slots:
                    date, appointment = random.choice(valid_slots)
                    _doctor_info[doctor]['schedule'][date].append(appointment)
                else:
                    date, appointment = None, None

            else:
                date, appointment = None, None
            
            follow_up_patient_info[name] = {
                'type': 'follow_up_visit',
                'department': department,
                'attending_physician': doctor,
                'date': date,
                'schedule': appointment,
                'symptom_level': symptom_level,
                'required_tests': combination,
                'gender': generate_random_code('gender'),
                'telecom': [{
                    'system': 'phone',
                    'value': generate_random_telecom(),
                    'use': generate_random_code('use')
                }],
                'birthDate': birth_date,
                'identifier': [{
                    'value': generate_random_id_number(birth_date=birth_date),
                    'use': 'official'
                }],
                'address': [{
                    'type': 'postal',
                    'text': generate_random_address(),
                    'use': 'home'
                }]
            }

        patient_info.update(follow_up_patient_info)
        merged_data = {
            'metadata': metadata,
            'department': department_info,
            'doctor': doctor_info,
            'test': fixed_test_schedule,
            'patient': patient_info,
        }

        return merged_data
    

    @staticmethod
    def test_list_generator(department: str,
                            min_n_per_department: int, 
                            max_n_per_department: int, 
                            file_path: Optional[str] = None) -> list[dict]:
        """
        Generate a list of test per department.
        
        Args:
            department (str): Target department.
            min_n_per_department (int): Minimum number of the test per department.
            max_n_per_department (int): Maximum number of the test per department.
            file_path (Optional[str], optional): Path to a file containing department and test information. 
                                                 If provided, it will be used to load names. Defaults to None.
        
        Returns:
            list[dict]: Dictionary list of tests per department.
        """
        if file_path == None:
            file_path = str(resources.files("h_adminsim.assets.departments").joinpath("department.json"))
        
        if registry.DEPARTMENT_TESTS is None:
            department_data = json_load(file_path)['specialty']
            registry.DEPARTMENT_TESTS = {
                k2: v2['tests']
                for v1 in department_data.values()
                for k2, v2 in v1['subspecialty'].items()
                if 'tests' in v2
            }
        
        all_tests = registry.DEPARTMENT_TESTS[department]
        code_to_test = {t['code']: t for t in all_tests}

        def _transitive_closure(code: str) -> set:
            """Return the full set of codes required by code (including itself)."""
            visited, stack = set(), [code]
            while stack:
                c = stack.pop()
                if c in visited:
                    continue
                visited.add(c)
                for dep in code_to_test.get(c, {}).get('depends_on', []):
                    stack.append(dep)
            return visited

        test_n = min(random.randint(min_n_per_department, max_n_per_department), len(all_tests))

        # Greedily pick tests in random order; add each test's full dep closure only
        # if it doesn't push the total over max_n_per_department.
        candidates = list(all_tests)
        random.shuffle(candidates)
        selected_codes = set()
        selected = list()

        for t in candidates:
            if len(selected_codes) >= test_n:
                break
            new_codes = sorted(list(_transitive_closure(t['code']) - selected_codes))
            if len(selected_codes) + len(new_codes) <= max_n_per_department:
                for c in new_codes:
                    selected_codes.add(c)
                    selected.append(code_to_test[c])

        # Sanity check: every depends_on must be present in the selected list
        final_codes = {t['code'] for t in selected}
        for t in selected:
            for dep in t.get('depends_on', []):
                assert dep in final_codes, \
                    log(
                    (
                        f"[test_list_generator] {t['code']} depends on {dep}, "
                        f"but {dep} is missing in selected tests for department '{department}'"
                    ), level='error'
                )

        if not (min_n_per_department <= len(selected) <= max_n_per_department):
            log(
                f"[test_list_generator] department '{department}': selected {len(selected)} tests, "
                f"expected [{min_n_per_department}, {max_n_per_department}]",
                level='warning'
            )

        return selected


    @staticmethod
    def _find_contiguous_slot(available_segments: list[int],
                              required_size: int) -> Optional[list[int]]:
        """
        Find a contiguous block of the required size from available segments.

        Args:
            available_segments (list[int]): Sorted list of available segment indices.
            required_size (int): Number of contiguous segments needed.

        Returns:
            Optional[list[int]]: A list of contiguous segment indices, or None if not found.
        """
        if len(available_segments) < required_size:
            return None

        available_sorted = sorted(available_segments)
        candidates = []

        current_block = [available_sorted[0]]
        for i in range(1, len(available_sorted)):
            if available_sorted[i] == available_sorted[i - 1] + 1:
                current_block.append(available_sorted[i])
            else:
                if len(current_block) >= required_size:
                    candidates.append(current_block)
                current_block = [available_sorted[i]]
        if len(current_block) >= required_size:
            candidates.append(current_block)

        if not candidates:
            return None

        block = random.choice(candidates)
        max_start = len(block) - required_size
        start_idx = random.randint(0, max_start)
        return block[start_idx:start_idx + required_size]


    @staticmethod
    def _sanity_check(merged_data: dict):
        """
        Validate merged data consistency.

        Args:
            merged_data (dict): Merged hospital data with both visit types.

        Raises:
            AssertionError: If validation fails.
        """
        for patient_name, pdata in merged_data['patient'].items():
            assert pdata.get('type') in ('first_visit', 'follow_up_visit'), \
                colorstr('red', f'Patient {patient_name} has invalid type: {pdata.get("type")}')
            assert pdata['department'] in merged_data['department'], \
                colorstr('red', f'Patient {patient_name} has invalid department {pdata["department"]}')
            assert pdata['attending_physician'] in merged_data['doctor'], \
                colorstr('red', f'Patient {patient_name} has invalid physician {pdata["attending_physician"]}')

            if pdata['type'] == 'follow_up_visit':
                assert len(pdata['required_tests']) > 0, \
                    colorstr('red', f'Follow-up patient {patient_name} has no required tests')
                for test in pdata['required_tests']:
                    assert 'test_code' in test, \
                        colorstr('red', f'Patient {patient_name} has test with missing fields')
                    assert len(test['schedule']) == 2, \
                        colorstr('red', f'Patient {patient_name} has test with invalid schedule')
                    assert 'priority' in test and isinstance(test['priority'], int) and test['priority'] >= 0, \
                        colorstr('red', f'Patient {patient_name} has test with invalid priority')

                # Validate sequential ordering and non-decreasing priority
                tests = pdata['required_tests']
                for i in range(len(tests) - 1):
                    t1, t2 = tests[i], tests[i + 1]

                    assert t1['priority'] <= t2['priority'], \
                        colorstr('red', f'Patient {patient_name}: priority decreases from test {i} ({t1["priority"]}) to test {i+1} ({t2["priority"]})')

                    t1_end_iso = get_iso_time(t1['schedule'][1], t1['date'])
                    t2_start_iso = get_iso_time(t2['schedule'][0], t2['date'])
                    assert not compare_iso_time(t1_end_iso, t2_start_iso), \
                        colorstr('red', f'Patient {patient_name}: test {i+1} starts ({t2_start_iso}) before test {i} ends ({t1_end_iso})')
                    
                # Validate consultation time
                if pdata['date'] is not None:
                    last_test_iso_time = get_iso_time(tests[-1]['schedule'][1], tests[-1]['date'])
                    consultation_time = get_iso_time(pdata['schedule'][0], pdata['date'])
                    assert not compare_iso_time(last_test_iso_time, consultation_time), \
                        colorstr('red', f'Patient {patient_name}: consultation starts ({consultation_time}) before last test ends ({last_test_iso_time})')
