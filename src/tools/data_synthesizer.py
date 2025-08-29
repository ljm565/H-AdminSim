import os
import random
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple

import registry
from tasks import ScheduleAssigner
from utils import Information, log, colorstr
from utils.common_utils import *
from utils.filesys_utils import json_load, txt_load, yaml_save, make_project_dir, json_save_fast
from utils.random_utils import (
    generate_random_prob,
    generate_random_date,
    generate_random_code,
    generate_random_names,
    generate_random_telecom,
    generate_random_specialty,
    generate_random_code_with_prob,
)



class DataSynthesizer:
    def __init__(self, config):
        # Initialize configuration, path and save directory
        self.config = config
        self._n = self.config.hospital_data.hospital_n
        self._save_dir = make_project_dir(self.config)
        self._data_save_dir = self._save_dir / 'data'
        yaml_save(self._save_dir / 'args.yaml', self.config)
        os.makedirs(self._data_save_dir, exist_ok=True)
        
    
    def synthesize(self,
                   return_obj: bool = False,
                   sanity_check: bool = False) -> Tuple[list[Information], list[Hospital]]:
        """
        Synthesize hospital data based on the configuration settings.

        Args:
            return_obj (bool, optional): Whether to return the hospital data object.
            sanity_check (bool, optional): If you want to check whether the generated data are compatible with the `Hospital` object,
                                 you can use this option.

        Raises:
            e: Exception if data synthesis fails.

        Returns:
            Tuple[list[Information], list[Hospital]]: A tuple containing the synthesized hospital data as an Information object and a Hospital object.
        """
        if sanity_check:
            return_obj = True

        try:
            all_data, all_hospitals = list(), list()
            hospitals = DataSynthesizer.hospital_list_generator(self.config.hospital_data.hospital_n)
            for i, hospital in tqdm(enumerate(hospitals), desc='Synthesizing data', total=len(hospitals)):
                data = DataSynthesizer.define_hospital_info(self.config, hospital)
                hospital_obj = convert_info_to_obj(data) if return_obj else None
                if sanity_check:
                    new_data = convert_obj_to_info(hospital_obj)
                    assert to_dict(data) == to_dict(new_data)
                json_save_fast(self._data_save_dir / f'hospital_{padded_int(i, len(str(self._n)))}.json', to_dict(data))
                all_data.append(data)
                all_hospitals.append(hospital_obj)
            log(f"Total {len(hospitals)} data synthesizing completed. Path: `{self._data_save_dir}`", color=True)
            return all_data, all_hospitals
        
        except Exception as e:
            log(f"Data synthesizing failed: {e}", level='error')
            raise e


    @staticmethod
    def define_hospital_info(config, hospital_name: str) -> Information:
        """
        Define the synthetic hospital data, including its departments and doctors.

        Args:
            config: Configuration object containing hospital data settings.
            hospital_name (str): Name of the hospital to be defined.

        Returns:
            Information: Synthetic data about the hospital.
        """
        # Define hosptial metadata
        days = config.hospital_data.days
        dates = generate_date_range(
            generate_random_iso_date_between(
                str(config.hospital_data.start_date.min),
                str(config.hospital_data.start_date.max),
            ), 
            days
        )
        interval_hour = float(config.hospital_data.interval_hour)
        start_hour = float(random.randint(config.hospital_data.start_hour.min, config.hospital_data.start_hour.max))
        end_hour = float(random.randint(config.hospital_data.end_hour.min, config.hospital_data.end_hour.max))
        operation_hour_per_day = int(end_hour - start_hour)
        department_n = random.randint(
            config.hospital_data.department_per_hospital.min,
            config.hospital_data.department_per_hospital.max
        )
        doctor_n_per_department = [random.randint(config.hospital_data.doctor_per_department.min, config.hospital_data.doctor_per_department.max) 
                                   for _ in range(department_n)]
        doctor_n = sum(doctor_n_per_department)
        doctor_capacity_per_hour_list = [c for c in range(config.hospital_data.doctor_capacity_per_hour.min, config.hospital_data.doctor_capacity_per_hour.max + 1) \
                                         if 1/c % interval_hour == 0]
        hospital_time_segments = convert_time_to_segment(start_hour, end_hour, interval_hour)
        metadata = Information(
            hospital_name=hospital_name,
            start_date=dates[0],
            end_date=dates[-1],
            days=days,
            department_num=department_n,
            doctor_num=doctor_n,
            time=Information(
                start_hour=start_hour,
                end_hour=end_hour,
                interval_hour=interval_hour
            )
        )

        # Define ScheduleAssigner class to randomly assign schedules to each doctor
        scheduler = ScheduleAssigner(start_hour, end_hour, interval_hour)

        # Define detailed hospital department, doctoral, and patient information
        department_info, doctor_info, patient_info = dict(), dict(), dict()
        departments = DataSynthesizer.department_list_generator(department_n)
        doctors = DataSynthesizer.name_list_generator(doctor_n, prefix='Dr. ')   # Doctor names are unique across all departments
        for department_data, doc_n in zip(departments, doctor_n_per_department):
            department, dep_code = department_data

            # Add department information
            department_info[department] = {'code': dep_code if dep_code else 'NA', 'doctor': []}
            
            # Add doctor information
            for _ in range(doc_n):
                doctor = doctors.pop()
                department_info[department]['doctor'].append(doctor)
                specialty, spe_code = generate_random_specialty(department)
                capacity_per_hour = random.choice(doctor_capacity_per_hour_list)
                working_days = random.randint(
                    config.hospital_data.working_days.min,
                    config.hospital_data.working_days.max
                )
                working_dates = sorted(random.sample(dates, working_days))
                doctor_info[doctor] = {
                    'department': department,
                    'specialty': {
                        'name': specialty,
                        'code': spe_code,
                    },
                    'schedule': {},
                    'capacity_per_hour': int(capacity_per_hour),
                    'capacity': int(capacity_per_hour * operation_hour_per_day * len(working_dates)),
                    'workload': 0.0,
                    'gender': generate_random_code('gender'),
                    'telecom': [{
                        'system': 'phone',
                        'value': generate_random_telecom(),
                        'use': generate_random_code('use')
                    }],
                    'birthDate': generate_random_date()
                }

                # Generate doctor schedules and apponitments based on the pre-defined days
                for date in dates:
                    # Working day case
                    if date in working_dates:
                        schedule_segments, schedule_times = scheduler(
                            generate_random_prob(
                                config.hospital_data.doctor_has_schedule_prob,
                                config.hospital_data.schedule_coverage_ratio.min,
                                config.hospital_data.schedule_coverage_ratio.max
                            )
                        )
                        doctor_info[doctor]['schedule'][date] = schedule_times
                    # Not working day case
                    else:
                        schedule_segments, schedule_times = scheduler(1)
                        doctor_info[doctor]['schedule'][date] = schedule_times

                    # Add patient information per doctor
                    patient_segments = list(set(hospital_time_segments) - set(sum(schedule_segments, [])))
                    _, appointments = scheduler(
                        generate_random_prob(
                            1,
                            config.hospital_data.appointment_coverage_ratio.min,
                            config.hospital_data.appointment_coverage_ratio.max
                        ),
                        True,
                        patient_segments,
                        max_chunk_size=1
                    )
                    patients = DataSynthesizer.name_list_generator(len(appointments))
                    for patient, appointment in zip(patients, appointments):
                        preference = generate_random_code_with_prob(
                            config.hospital_data.preference.type,
                            config.hospital_data.preference.probs
                        )
                        symptom_level = generate_random_code_with_prob(
                            config.hospital_data.symptom.type,
                            config.hospital_data.symptom.probs
                        )
                        patient_info[patient] = {
                            'department': department,
                            'attending_physician': doctor,
                            'date': date,
                            'schedule': appointment,
                            'preference': preference,
                            'symptom_level': symptom_level,
                            'gender': generate_random_code('gender'),
                            'telecom': [{
                                'system': 'phone',
                                'value': generate_random_telecom(),
                                'use': generate_random_code('use')
                            }],
                            'birthDate': generate_random_date()
                        }
            
        # Finalize data structure
        data = Information(
            metadata=metadata,
            department=department_info,
            doctor=doctor_info,
            patient=patient_info,
        )

        # Data sanity check
        if len(data.department) != metadata.department_num:
            raise AssertionError(colorstr('red', 'Department number mismatch'))
        if len(data.department) != len(set(doc['department'] for doc in data.doctor.values())):
            raise AssertionError(colorstr('red', 'Department number mismatch'))
        if len(data.doctor) != metadata.doctor_num:
            raise AssertionError(colorstr('red', 'Doctor number mismatch'))
        if len(data.doctor) != sum(len(dept['doctor']) for dept in data.department.values()):
            raise AssertionError(colorstr('red', 'Doctor number mismatch'))
        
        return data


    @staticmethod
    def hospital_list_generator(hospital_n: int,
                                file_path: Optional[str] = None) -> list[str]:
        """
        Generate a list of hospital names based on the number of hospitals.
        
        Args:
            hospital_n (int): Number of hospitals to generate.
            file_path (Optional[str], optional): Path to a file containing hospital names. If provided, it will be used to load names.
        
        Returns:
            list[str]: List of hospital names in the format "Hospital 001", "Hospital 002", etc.
        """
        if file_path:
            if registry.HOSPITALS is None:
                registry.HOSPITALS = [word.capitalize() for word in txt_load(file_path).split('\n') if word.strip()]
            return [f"{random.choice(registry.HOSPITALS)}" for _ in range(hospital_n)]
        
        zfill_l = len(str(hospital_n))
        return [f"hospital_{padded_int(i, zfill_l)}" for i in range(hospital_n)]

    
    @staticmethod
    def department_list_generator(department_n: int,
                                  file_path: Optional[str] = 'asset/departments/department.json') -> list[Tuple[str, str]]:
        """
        Generate a list of department names based on the number of departments.
        
        Args:
            department_n (int): Number of departments to generate.
            file_path (Optional[str], optional): Path to a file containing department names. If provided, it will be used to load names.
                                                 Defaults to 'asset/departments/department.json'.
        
        Returns:
            list[Tuple[str, str]]: List of department names and their codes.
        """
        if file_path:
            if registry.DEPARTMENTS is None:
                specialty = json_load(file_path)['specialty']
                registry.DEPARTMENTS = [(k2, v2['code']) for v1 in specialty.values() for k2, v2 in v1['subspecialty'].items()]
            
            if department_n > len(registry.DEPARTMENTS):
                raise ValueError(f"Requested {department_n} departments, but only {len(registry.DEPARTMENTS)} available in {file_path}.")
        
            return random.sample(registry.DEPARTMENTS, department_n)
            
        zfill_l = len(str(department_n))
        return [(f"department_{padded_int(i, zfill_l)}", None) for i in range(department_n)]
    
    
    @staticmethod
    def name_list_generator(n: int,
                            first_name_file_path: str = 'asset/names/firstname.txt', 
                            last_name_file_path: str = 'asset/names/lastname.txt',
                            prefix: Optional[str] = None) -> list[str]:
        """
        Generate a list of names.
        
        Args:
            n (int): Number of doctors to generate.
            first_name_file_path (str, optional): Path to a file containing first names. Defaults to 'asset/names/firstname.txt'.
            last_name_file_path (str, optional): Path to a file containing last names. Defaults to 'asset/names/lastname.txt'.
            prefix (Optional[str], optional): Prefix for to be generated names.
        
        Returns:
            list[str]: List of names.
        """
        if prefix != None:
            assert isinstance(prefix, str), log("`prefix` must be a string type", "error")
            names = [f'{prefix}{name}' for name in generate_random_names(n, first_name_file_path, last_name_file_path)]
        else:
            names = [name for name in generate_random_names(n, first_name_file_path, last_name_file_path)]
        random.shuffle(names)
        return names
