from tqdm import tqdm
from typing import Tuple
from decimal import Decimal, getcontext

from .data_synthesizer import DataSynthesizer
from h_adminsim.task.schedule_assign import ScheduleAssigner
from h_adminsim.utils import Information, log
from h_adminsim.utils.common_utils import *
from h_adminsim.utils.filesys_utils import *
from h_adminsim.utils.random_utils import (
    generate_random_prob,
    generate_random_date,
    generate_random_code,
    generate_random_address,
    generate_random_telecom,
    generate_random_id_number,
    generate_random_occupation,
    generate_random_code_with_prob,
)



class FirstVisitDataSynthesizer(DataSynthesizer):
    def __init__(self, config):
        super().__init__(config)
        getcontext().prec = 10


    def synthesize(self, department_info_path: Optional[str] = None) -> list[Information]:
        """
        Synthesize hospital data based on the configuration settings.

        Args:
            department_info_path (Optional[str], optional): Path to a file containing department information. If provided, it will be used to load names. 
                                                            Defaults to None.

        Raises:
            e: Exception if data synthesis fails.

        Returns:
            list[Information]: A list containing the synthesized hospital data as Information objects.
        """
        try:
            all_data = list()
            hospitals = DataSynthesizer.hospital_list_generator(self.config.hospital_data.hospital_n)
            for i, hospital in tqdm(enumerate(hospitals), desc='Synthesizing data..', total=len(hospitals)):
                data = DataSynthesizer.define_hospital_info(self.config, hospital, department_info_path)
                data = FirstVisitDataSynthesizer.generate_first_visit_patients(self.config, data)
                json_save_fast(self.data_save_dir / f'hospital_{padded_int(i, len(str(self._n)))}.json', to_dict(data))
                all_data.append(data)
            log(f"Total {len(hospitals)} data synthesizing completed. Path: `{self.data_save_dir}`", color=True)
            return all_data

        except Exception as e:
            log(f"Data synthesizing failed: {e}", level='error')
            raise


    @staticmethod
    def generate_first_visit_patients(config, data: Information) -> Information:
        """
        Generate first-visit patient profiles and add them to the hospital data.

        Iterates over each doctor and their schedule dates to compute available
        patient appointment slots, then generates patient profiles with first-visit
        specific fields (symptom_level, preference).

        Args:
            config: Configuration object with hospital_data.first_visit settings.
            data (Information): Hospital data (metadata, department, doctor, patient={})
                                as returned by DataSynthesizer.define_hospital_info().

        Returns:
            Information: Hospital data with first-visit patients populated.
        """
        # Extract time parameters
        visit_type = 'first_visit'
        start_hour = float(data.metadata.time.start_hour)
        end_hour = float(data.metadata.time.end_hour)
        interval_hour = float(data.metadata.time.interval_hour)
        hospital_time_segments = convert_time_to_segment(start_hour, end_hour, interval_hour)

        # Build scheduler
        scheduler = ScheduleAssigner(start_hour, end_hour, interval_hour)

        patient_info = {}
        for doctor, doc_data in data.doctor.items():
            department = doc_data['department']
            capacity_per_hour = doc_data['capacity_per_hour']
            duration = int(Decimal(str(1)) / Decimal(str(capacity_per_hour)) / Decimal(str(interval_hour)))

            for date, schedule_times in doc_data['schedule'].items():
                # Recompute schedule segments from stored schedule_times
                schedule_segments_flat = []
                for time_range in schedule_times:
                    schedule_segments_flat.extend(
                        convert_time_to_segment(start_hour, end_hour, interval_hour, time_range=time_range)
                    )

                # Compute available patient segments
                vacant_segments = list(set(hospital_time_segments) - set(schedule_segments_flat))

                # Generate appointments from available segments
                _, appointments = scheduler(
                    generate_random_prob(
                        1,
                        config.hospital_data.appointment_coverage_ratio.min,
                        config.hospital_data.appointment_coverage_ratio.max
                    ),
                    True,
                    vacant_segments,
                    min_chunk_size=duration,
                    max_chunk_size=duration
                )

                # Generate patient profiles
                patients = DataSynthesizer.name_list_generator(len(appointments))
                for patient, appointment in zip(patients, appointments):
                    preference = generate_random_code_with_prob(
                        config.hospital_data.first_visit.preference.type,
                        config.hospital_data.first_visit.preference.probs
                    )
                    preference_rank = DataSynthesizer.second_preference_generator(preference, visit_type)
                    symptom_level = generate_random_code_with_prob(
                        config.hospital_data.first_visit.symptom.type,
                        config.hospital_data.first_visit.symptom.probs
                    )
                    birth_date = generate_random_date()
                    patient_info[patient] = [{
                        'visit_type': visit_type,
                        'department': department,
                        'attending_physician': doctor,
                        'date': date,
                        'schedule': appointment,
                        'preference': preference_rank,
                        'symptom_level': symptom_level,
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
                        }],
                        'occupation': generate_random_occupation(),
                    }]

        data.patient = patient_info
        return data