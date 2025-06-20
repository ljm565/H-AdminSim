import os
import random
import numpy as np
from typing import Optional, Tuple

import registry
from registry import Hospital
from utils import Information, colorstr
from utils.common_utils import padded_int, to_dict
from utils.random_utils import generate_random_names
from utils.filesys_utils import txt_load, json_save, yaml_save, make_project_dir



class DataSynthesizer:
    def __init__(self, config):
        # Initialize configuration, path and save directory
        self._config = config        
        self._save_dir = make_project_dir(self._config)
        self._data_save_dir = self._save_dir / 'data'
        yaml_save(self._save_dir / 'args.yaml', self._config)
        os.makedirs(self._data_save_dir, exist_ok=True)
        
    
    def synthesize(self):
        hospitals = self.make_hospital(self._config.hospital_data.hospital_n)
        for i, hospital in enumerate(hospitals):
            data, hospital_obj = self.define_hospital_info(self._config, hospital)
            json_save(self._data_save_dir / f'hospital_{padded_int(i)}.json', to_dict(data))


    def define_hospital_info(self, config, hospital_name: str) -> Tuple[Information, Hospital]:
        """
        Define the metadata and structure of a hospital, including its departments and doctors.

        Args:
            config (Config): Configuration object containing hospital data settings.
            hospital_name (str): Name of the hospital to be defined.

        Returns:
            Tuple[Information, Hospital]: Metadata about the hospital and a Hospital object containing its structure.
        """
        # Define hosptial metadata
        hospital_obj = Hospital(hospital_name)
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
            docotor_num=doctor_n,
            time=Information(
                start_hour=start_hour,
                end_hour=end_hour,
                inteveal_hour=interval_hour
            )
        )

        # Define detailed hospital department and doctoral information
        department_info, doctor_info = dict(), dict()
        departments = self.make_departments(department_n)
        doctors = self.make_doctors(doctor_n)   # Doctor names are unique across all departments
        for department, doc_n in zip(departments, doctor_n_per_department):
            # Add department to hospital
            department_info[department] = {'doctor': []}
            department_obj = hospital_obj.add_department(department)
            
            # Add doctors to department
            for _ in range(doc_n):
                doctor = doctors.pop()
                department_obj.add_doctor(doctor)
                department_info[department]['doctor'].append(doctor)
                doctor_info[doctor] = {
                    'department': department,
                    'schedule': 0
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
        if len(data.doctor) != metadata.docotor_num:
            raise AssertionError(colorstr('red', 'Doctor number mismatch'))
        if len(data.doctor) != sum(len(dept['doctor']) for dept in data.department.values()):
            raise AssertionError(colorstr('red', 'Doctor number mismatch'))
        
        return data, hospital_obj


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
        return [f"hospital_{padded_int(i+1)}" for i in range(hospital_n)]

    
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
        return [f"department_{padded_int(i+1)}" for i in range(department_n)]
    

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
        return [f'Dr. {name}' for name in generate_random_names(doctor_n, first_name_file_path, last_name_file_path)]
