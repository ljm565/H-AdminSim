import random
from tqdm import tqdm
from typing import Optional
from decimal import getcontext

from h_adminsim.task.schedule_assign import ScheduleAssigner
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.common_utils import (
    convert_time_to_segment,
    convert_segment_to_time,
)
from h_adminsim.utils.filesys_utils import *
from h_adminsim.utils.random_utils import (
    generate_random_code,
    generate_random_date,
    generate_random_tests,
    generate_random_telecom,
    generate_random_symptom,
    generate_random_address,
    generate_random_id_number,
    generate_random_code_with_prob,    
)
from h_adminsim.tools.synthdat.data_synthesizer import DataSynthesizer



class FollowUpDataSynthesizer(DataSynthesizer):
    def __init__(self, config, source_data_dir: str):
        super().__init__(config)
        self.source_data_dir = source_data_dir
        self.source_data_files = sorted(get_files(source_data_dir, ext='json'))
        getcontext().prec = 10

        if not self.source_data_files:
            raise FileNotFoundError(
                log(f"No hospital JSON files found in {source_data_dir}", "error")
            )


    def synthesize(self, sanity_check: bool = False) -> list[dict]:
        """
        Synthesize follow-up patients and merge them into existing hospital data files.

        For each hospital_*.json, generates follow-up patients and adds them to
        the existing patient dict (alongside first_visit patients), then overwrites
        the same file.

        Args:
            sanity_check (bool, optional): Whether to validate generated data. Defaults to False.

        Returns:
            list[dict]: List of merged hospital data dicts (containing both first_visit and follow_up_visit patients).
        """
        try:
            all_data = []
            for data_file in tqdm(
                self.source_data_files,
                desc='Synthesizing follow-up patient data..',
                total=len(self.source_data_files)
            ):
                hospital_data = json_load(data_file)
                merged_data = FollowUpDataSynthesizer.generate_followup_patients(
                    self.config, hospital_data
                )

                if sanity_check:
                    FollowUpDataSynthesizer._sanity_check(merged_data)

                # Overwrite the same file with merged data
                json_save_fast(data_file, merged_data)
                all_data.append(merged_data)

            log(f"Total {len(all_data)} follow-up data merged into existing files. Path: `{self.source_data_dir}`", color=True)
            return all_data

        except Exception as e:
            log(f"Follow-up data synthesizing failed: {e}", level='error')
            raise


    @staticmethod
    def generate_followup_patients(config, hospital_data: dict) -> dict:
        """
        Generate follow-up patient profiles and merge them into hospital data.

        Reads hospital metadata, departments, and doctor information from existing
        hospital data, generates follow-up patients who need medical tests scheduled,
        and merges them into the same patient dict.

        Args:
            config: Configuration object with hospital_data.follow_up_visit settings.
            hospital_data (dict): Existing hospital data containing
                                   metadata, department, doctor, and patient dicts.

        Returns:
            dict: Hospital data with follow-up patients merged into the patient dict.
        """
        metadata = hospital_data['metadata']
        department_info = hospital_data['department']
        doctor_info = hospital_data['doctor']
        patient_info = hospital_data.get('patient', {})

        # Hospital time parameters
        start_hour = float(metadata['time']['start_hour'])
        end_hour = float(metadata['time']['end_hour'])
        interval_hour = float(metadata['time']['interval_hour'])
        hospital_time_segments = convert_time_to_segment(start_hour, end_hour, interval_hour)

        # Follow-up config
        fu_config = config.hospital_data.follow_up_visit
        tests_min = fu_config.tests_per_patient.min
        tests_max = fu_config.tests_per_patient.max
        cross_dept_prob = fu_config.cross_department_test_prob
        include_consultation = fu_config.get('include_consultation', True)

        # Calculate follow-up patient count from prob ratio
        fu_prob = fu_config.prob
        existing_fv_count = sum(1 for p in patient_info.values() if p.get('type') == 'first_visit')
        followup_patient_count = max(1, round(existing_fv_count * fu_prob)) if existing_fv_count > 0 else 0

        if followup_patient_count == 0:
            return hospital_data

        # Build scheduler
        scheduler = ScheduleAssigner(start_hour, end_hour, interval_hour)

        # Collect occupied segments per doctor per date from existing patients
        occupied_segments = FollowUpDataSynthesizer._build_occupied_segments(
            doctor_info, patient_info, start_hour, end_hour, interval_hour
        )

        # Generate follow-up patients
        doctors_list = list(doctor_info.keys())
        names = DataSynthesizer.name_list_generator(followup_patient_count)

        for _ in range(followup_patient_count):
            if not names:
                break
            patient_name = names.pop()

            # Pick a random doctor (patient already has a department from prior visit)
            doctor = random.choice(doctors_list)
            doc_data = doctor_info[doctor]
            department = doc_data['department']
            capacity_per_hour = doc_data['capacity_per_hour']
            duration = int(1 / capacity_per_hour / interval_hour)

            # Find working dates for this doctor (dates with non-full schedules)
            working_dates = [
                date for date, sched in doc_data['schedule'].items()
                if len(sched) < len(hospital_time_segments)
            ]
            if not working_dates:
                continue

            # Select diagnosis from symptom.json
            disease_info = generate_random_symptom(department, ensure_unique_department=False, verbose=False)
            diagnosis = disease_info['disease'] if isinstance(disease_info, dict) else 'Unknown'

            # Select required tests
            n_tests = random.randint(tests_min, tests_max)
            required_tests = generate_random_tests(
                department, n_tests,
                cross_department_prob=cross_dept_prob
            )

            # Assign time slots for each test
            scheduled_tests = []
            for test in required_tests:
                test_duration_segments = max(1, int(test['duration_hour'] / interval_hour))

                # Pick a date for the test (from working dates)
                test_date = random.choice(working_dates)

                # Get all hospital segments
                doc_schedule_segments = convert_time_to_segment(
                    start_hour, end_hour, interval_hour,
                    time_range=None
                )
                doc_occupied = doc_data['schedule'].get(test_date, [])
                doc_occupied_seg = convert_time_to_segment(
                    start_hour, end_hour, interval_hour,
                    time_range=doc_occupied
                ) if doc_occupied and len(doc_occupied) == 2 else []

                # Already occupied by existing patients and previously assigned tests
                occ_key = (doctor, test_date)
                already_occupied = occupied_segments.get(occ_key, set())

                available = [
                    s for s in doc_schedule_segments
                    if s not in already_occupied and s not in set(doc_occupied_seg)
                ]

                # Try to find a contiguous block
                slot = FollowUpDataSynthesizer._find_contiguous_slot(
                    available, test_duration_segments
                )
                if slot is None:
                    continue

                # Mark as occupied
                for seg in slot:
                    already_occupied.add(seg)
                occupied_segments[occ_key] = already_occupied

                # Convert segments to time
                time_start, time_end = convert_segment_to_time(
                    start_hour, end_hour, interval_hour, slot
                )
                scheduled_tests.append({
                    'test_name': test['test_name'],
                    'test_code': test['test_code'],
                    'test_department': test['test_department'],
                    'duration_hour': test['duration_hour'],
                    'date': test_date,
                    'schedule': [time_start, time_end],
                })

            if not scheduled_tests:
                continue

            # Optional: schedule consultation after tests
            consultation = None
            if include_consultation:
                # Find the latest test date
                latest_test_date = max(t['date'] for t in scheduled_tests)
                latest_test_end_seg = None
                for t in scheduled_tests:
                    if t['date'] == latest_test_date:
                        end_seg = convert_time_to_segment(
                            start_hour, end_hour, interval_hour,
                            time_range=t['schedule']
                        )
                        if end_seg:
                            last = max(end_seg)
                            if latest_test_end_seg is None or last > latest_test_end_seg:
                                latest_test_end_seg = last

                # Find available slot after the latest test
                occ_key = (doctor, latest_test_date)
                already_occupied = occupied_segments.get(occ_key, set())
                doc_occupied = doc_data['schedule'].get(latest_test_date, [])
                doc_occupied_seg = convert_time_to_segment(
                    start_hour, end_hour, interval_hour,
                    time_range=doc_occupied
                ) if doc_occupied and len(doc_occupied) == 2 else []

                available_after = [
                    s for s in hospital_time_segments
                    if s not in already_occupied
                    and s not in set(doc_occupied_seg)
                    and (latest_test_end_seg is None or s > latest_test_end_seg)
                ]

                consult_slot = FollowUpDataSynthesizer._find_contiguous_slot(
                    available_after, duration
                )
                if consult_slot:
                    for seg in consult_slot:
                        already_occupied.add(seg)
                    occupied_segments[occ_key] = already_occupied

                    c_start, c_end = convert_segment_to_time(
                        start_hour, end_hour, interval_hour, consult_slot
                    )
                    consultation = {
                        'date': latest_test_date,
                        'schedule': [c_start, c_end],
                        'attending_physician': doctor,
                    }

            # Build patient profile
            preference = generate_random_code_with_prob(
                fu_config.preference.type,
                fu_config.preference.probs
            )
            preference_rank = DataSynthesizer.second_preference_generator(preference)
            birth_date = generate_random_date()

            patient_info[patient_name] = {
                'type': 'follow_up_visit',
                'department': department,
                'attending_physician': doctor,
                'diagnosis': diagnosis,
                'required_tests': scheduled_tests,
                'consultation': consultation,
                'preference': preference_rank,
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

        merged_data = {
            'metadata': metadata,
            'department': department_info,
            'doctor': doctor_info,
            'patient': patient_info,
        }

        return merged_data


    @staticmethod
    def _build_occupied_segments(doctor_info: dict,
                                 existing_patients: dict,
                                 start_hour: float,
                                 end_hour: float,
                                 interval_hour: float) -> dict:
        """
        Build a mapping of (doctor, date) -> set of occupied segment indices
        from existing patient appointments and doctor schedules.

        Args:
            doctor_info (dict): Doctor information dict.
            existing_patients (dict): Existing patient data.
            start_hour (float): Hospital opening hour.
            end_hour (float): Hospital closing hour.
            interval_hour (float): Time interval in hours.

        Returns:
            dict: Mapping of (doctor_name, date) -> set of occupied segment indices.
        """
        occupied = {}

        for patient_data in existing_patients.values():
            doctor = patient_data.get('attending_physician')
            date = patient_data.get('date')
            schedule = patient_data.get('schedule')
            if doctor and date and schedule and len(schedule) == 2:
                key = (doctor, date)
                segs = convert_time_to_segment(
                    start_hour, end_hour, interval_hour,
                    time_range=schedule
                )
                if key not in occupied:
                    occupied[key] = set()
                occupied[key].update(segs)

        return occupied


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
                    assert 'test_name' in test and 'test_code' in test, \
                        colorstr('red', f'Patient {patient_name} has test with missing fields')
                    assert len(test['schedule']) == 2, \
                        colorstr('red', f'Patient {patient_name} has test with invalid schedule')
