import os
import random
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple

import registry
from tasks import ScheduleAssigner
from utils import Information, log, colorstr
from utils.common_utils import *
from utils.random_utils import generate_random_names
from utils.filesys_utils import txt_load, json_save, yaml_save, make_project_dir



class DataSynthesizer:
    def __init__(self, config):
        # Initialize configuration, path and save directory
        self._config = config
        self._n = self._config.hospital_data.hospital_n
        self._save_dir = make_project_dir(self._config)
        self._data_save_dir = self._save_dir / 'data'
        yaml_save(self._save_dir / 'args.yaml', self._config)
        os.makedirs(self._data_save_dir, exist_ok=True)
        
    
    def synthesize(self) -> Tuple[Information, Hospital]:
        """
        Synthesize hospital data based on the configuration settings.

        Raises:
            e: Exception if data synthesis fails.

        Returns:
            Tuple[Information, Hospital]: A tuple containing the synthesized hospital data as an Information object and a Hospital object.
        """
        try:
            hospitals = self.make_hospital(self._config.hospital_data.hospital_n)
            for i, hospital in tqdm(enumerate(hospitals), desc='Synthesizing data', total=len(hospitals)):
                data = self.define_hospital_info(self._config, hospital)
                hospital_obj = convert_info_to_obj(data)
                json_save(self._data_save_dir / f'hospital_{padded_int(i, len(str(self._n)))}.json', to_dict(data))
            log(f"Total {len(hospitals)} data synthesizing completed. Path: `{self._data_save_dir}`", color=True)
            return data, hospital_obj
        
        except Exception as e:
            log(f"Data synthesizing failed: {e}", level='error')
            raise e


    def define_hospital_info(self, config, hospital_name: str) -> Information:
        """
        Define the synthetic hospital data, including its departments and doctors.

        Args:
            config (Config): Configuration object containing hospital data settings.
            hospital_name (str): Name of the hospital to be defined.

        Returns:
            Information: Synthetic data about the hospital.
        """
        # Define hosptial metadata
        interval_hour = config.hospital_data.interval_hour
        start_hour = random.choice(
            np.arange(
                config.hospital_data.start_hour.min,
                config.hospital_data.start_hour.max+interval_hour,  # Ensure the end hour is inclusive
                interval_hour
            )
        )
        end_hour = random.choice(
            np.arange(
                config.hospital_data.end_hour.min,
                config.hospital_data.end_hour.max+interval_hour,  # Ensure the end hour is inclusive
                interval_hour
            )
        )
        department_n = random.randint(
            config.hospital_data.department_per_hospital.min,
            config.hospital_data.department_per_hospital.max
        )
        doctor_n_per_department = [random.randint(config.hospital_data.doctor_per_department.min, config.hospital_data.doctor_per_department.max) 
                                   for _ in range(department_n)]
        doctor_n = sum(doctor_n_per_department)

        metadata = Information(
            hospital_name=hospital_name,
            department_num=department_n,
            doctor_num=doctor_n,
            time=Information(
                start_hour=start_hour,
                end_hour=end_hour,
                inteveal_hour=interval_hour
            )
        )

        # Define SchedulerAssigner class to randomly assign schedules to each doctor
        scheduler = ScheduleAssigner(start_hour, end_hour, interval_hour)

        # Define detailed hospital department and doctoral information
        department_info, doctor_info = dict(), dict()
        departments = self.make_departments(department_n)
        doctors = self.make_doctors(doctor_n)   # Doctor names are unique across all departments
        for department, doc_n in zip(departments, doctor_n_per_department):
            # Add department to hospital
            department_info[department] = {'doctor': []}
            
            # Add doctors to department
            for _ in range(doc_n):
                doctor = doctors.pop()
                department_info[department]['doctor'].append(doctor)
                doctor_info[doctor] = {
                    'department': department,
                    'schedule': scheduler(
                        self.random_prob(
                            config.hospital_data.doctor_has_schedule_prob,
                            config.hospital_data.schedule_coverage_ratio.min,
                            config.hospital_data.schedule_coverage_ratio.max
                        )
                    )
                }
            
        # Finalize data structure
        data = Information(
            metadata=metadata,
            department=department_info,
            doctor=doctor_info,
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


    def make_hospital(self, hospital_n: int, file_path: Optional[str] = None) -> list[str]:
        """
        Generate a list of hospital names based on the number of hospitals.
        
        Args:
            hospital_n (int): Number of hospitals to generate.
            file_path (Optional[str]): Path to a file containing hospital names. If provided, it will be used to load names.
        
        Returns:
            list[str]: List of hospital names in the format "Hospital 001", "Hospital 002", etc.
        """
        if file_path:
            if registry.HOSPITALS is None:
                registry.HOSPITALS = [word.capitalize() for word in txt_load(file_path).split('\n')]
            return [f"{random.choice(registry.HOSPITALS)}" for _ in range(hospital_n)]
        return [f"hospital_{padded_int(i, len(str(self._n)))}" for i in range(hospital_n)]

    
    def make_departments(self, department_n: int, file_path: Optional[str] = None) -> list[str]:
        """
        Generate a list of department names based on the number of departments.
        
        Args:
            department_n (int): Number of departments to generate.
            file_path (Optional[str]): Path to a file containing department names. If provided, it will be used to load names.
        
        Returns:
            list[str]: List of department names in the format "Department 001", "Department 002", etc.
        """
        if file_path:
            if registry.DEPARTMENTS is None:
                registry.DEPARTMENTS = [word.capitalize() for word in txt_load(file_path).split('\n')]
            return [f"{random.choice(registry.DEPARTMENTS)}" for _ in range(department_n)]
        return [f"department_{padded_int(i, len(str(self._n)))}" for i in range(department_n)]
    

    def make_doctors(self,
                     doctor_n: int,
                     first_name_file_path: str = 'asset/names/firstname.txt',
                     last_name_file_path: str = 'asset/names/lastname.txt') -> list[str]:
        """
        Generate a list of doctor names based on the number of doctors.
        
        Args:
            doctor_n (int): Number of doctors to generate.
            first_name_file_path (str): Path to a file containing first names. Defaults to 'asset/names/firstname.txt'.
            last_name_file_path (str): Path to a file containing last names. Defaults to 'asset/names/lastname.txt'.
        
        Returns:
            list[str]: List of doctor names in the format "Doctor 001", "Doctor 002", etc.
        """
        doctors = [f'Dr. {name}' for name in generate_random_names(doctor_n, first_name_file_path, last_name_file_path)]
        random.shuffle(doctors)
        return doctors
    

    def random_prob(self, 
                    doctor_has_schedule_prob: float,
                    coverage_min: float,
                    coverage_max: float) -> float:
        """
        Determine the final schedule ratio for a doctor.

        Args:
            doctor_has_schedule_prob (float): Probability that a doctor has any schedule.
            coverage_min (float): Minimum proportion of total available hours the schedule can occupy.
            coverage_max (float): Maximum proportion of total available hours the schedule can occupy.

        Returns:
            float: The final schedule ratio. 0.0 if the doctor has no schedule. A float in [coverage_min, coverage_max] if the doctor has a schedule.
        """
        has_schedule = random.random() < doctor_has_schedule_prob
        if not has_schedule:
            return 0.0

        return random.uniform(coverage_min, coverage_max)
