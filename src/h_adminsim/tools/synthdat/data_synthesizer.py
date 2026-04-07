import os
import random
from importlib import resources
from typing import Optional, Tuple
from decimal import Decimal, getcontext

from h_adminsim.task.schedule_assign import ScheduleAssigner
from h_adminsim.utils import Information, log, colorstr
from h_adminsim.utils.common_utils import *
from h_adminsim.utils.filesys_utils import *
from h_adminsim.utils.random_utils import (
    generate_random_prob,
    generate_random_date,
    generate_random_code,
    generate_random_names,
    generate_random_telecom,
    generate_random_specialty,
)



class DataSynthesizer:
    def __init__(self, config, existint_data_dir: Optional[str] = None):
        # Initialize configuration, path and save directory
        self.config = config
        self._n = self.config.hospital_data.hospital_n
        if existint_data_dir is None:
            self.save_dir = make_project_dir(self.config)
            self.data_save_dir = self.save_dir / 'data'
            yaml_save(self.save_dir / 'args.yaml', self.config)
            os.makedirs(self.data_save_dir, exist_ok=True)
        else:
            self.data_save_dir = existint_data_dir
            os.makedirs(self.data_save_dir, exist_ok=True)
        getcontext().prec = 10
        

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
                                         if float(Decimal(str(1))/Decimal(str(c)) % Decimal(str(interval_hour))) == 0]
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

        # Define detailed hospital department and doctoral information
        department_info, doctor_info = dict(), dict()
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
                    'gender': generate_random_code('gender'),
                    'telecom': [{
                        'system': 'phone',
                        'value': generate_random_telecom(),
                        'use': generate_random_code('use')
                    }],
                    'birthDate': generate_random_date()
                }
                # Generate doctor schedules based on the pre-defined days
                for date in dates:
                    # Working day case
                    if date in working_dates:
                        _, schedule_times = scheduler(
                            generate_random_prob(
                                config.hospital_data.doctor_has_schedule_prob,
                                config.hospital_data.schedule_coverage_ratio.min,
                                config.hospital_data.schedule_coverage_ratio.max
                            )
                        )
                        doctor_info[doctor]['schedule'][date] = schedule_times
                    # Not working day case
                    else:
                        _, schedule_times = scheduler(1)
                        doctor_info[doctor]['schedule'][date] = schedule_times

        # Finalize data structure
        data = Information(
            metadata=metadata,
            department=department_info,
            doctor=doctor_info,
            patient={},
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
                                  file_path: Optional[str] = None) -> list[Tuple[str, str]]:
        """
        Generate a list of department names based on the number of departments.
        
        Args:
            department_n (int): Number of departments to generate.
            file_path (Optional[str], optional): Path to a file containing department names. If provided, it will be used to load names. Defaults to None.
        
        Returns:
            list[Tuple[str, str]]: List of department names and their codes.
        """
        if file_path == None:
            file_path = str(resources.files("h_adminsim.assets.departments").joinpath("department.json"))

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
                            first_name_file_path: Optional[str] = None, 
                            last_name_file_path: Optional[str] = None,
                            prefix: Optional[str] = None,
                            reject_list: Optional[list[str]] = None) -> list[str]:
        """
        Generate a list of names.
        
        Args:
            n (int): Number of doctors to generate.
            first_name_file_path (Optional[str], optional): Path to a file containing first names. Defaults to None.
            last_name_file_path (Optional[str], optional): Path to a file containing last names. Defaults to None.
            prefix (Optional[str], optional): Prefix for to be generated names.
            reject_list (Optional[list[str]], optional): List of names to exclude.
        
        Returns:
            list[str]: List of names.
        """
        if first_name_file_path == None:
            first_name_file_path = str(resources.files("h_adminsim.assets.names").joinpath("firstname.txt"))
        if last_name_file_path == None:
            last_name_file_path = str(resources.files("h_adminsim.assets.names").joinpath("lastname.txt"))

        if prefix != None:
            assert isinstance(prefix, str), log("`prefix` must be a string type", "error")
            names = [f'{prefix}{name}' for name in generate_random_names(n, first_name_file_path, last_name_file_path, reject_list)]
        else:
            names = [name for name in generate_random_names(n, first_name_file_path, last_name_file_path, reject_list)]
        random.shuffle(names)
        return names


    @staticmethod
    def second_preference_generator(preference: str, visit_type: str) -> list[str]:
        """
        Generate a list of preferences based on the initial preference.

        Args:
            preference (str): First priority of preference.
            visit_type (str): Hospital visit type. Currently support [`first_visit`, `follow_up_visit`]

        Returns:
            list[str]: List of preferences including first and second priority.
        """
        preference_list = [preference]
        
        if visit_type == 'first_visit':
            if preference == 'doctor':
                second_preference = random.choice(['asap', 'date'])
                preference_list.append(second_preference)
            elif preference == 'date':
                second_preference = random.choice(['asap', 'doctor'])
                preference_list.append(second_preference)
            elif preference == 'asap':
                second_preference = random.choice(['date', 'doctor'])
                preference_list.append(second_preference)
        
        elif visit_type == 'follow_up_visit':
            if preference == 'asap':
                second_preference = random.choice(['batch'])
                preference_list.append(second_preference)
            elif preference == 'batch':
                second_preference = random.choice(['asap'])
                preference_list.append(second_preference)

        return preference_list
