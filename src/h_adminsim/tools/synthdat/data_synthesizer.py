import os
import random
from datetime import timedelta
from importlib import resources
from decimal import Decimal, getcontext
from typing import Optional, TYPE_CHECKING

from h_adminsim.registry import variables
from h_adminsim.task.schedule_assign import ScheduleAssigner
from h_adminsim.utils import Information, colorstr
from h_adminsim.utils.common_utils import (
    padded_int,
    str_to_datetime,
    datetime_to_str,
    generate_date_range,
    generate_random_iso_date_between,
)
from h_adminsim.utils.filesys_utils import (
    txt_load,
    yaml_save,
    json_load,
    make_project_dir,
)
from h_adminsim.utils.random_utils import (
    generate_random_prob,
    generate_random_date,
    generate_random_code,
    generate_random_names,
    generate_random_telecom,
    generate_random_specialty,
)

if TYPE_CHECKING:
    from sconf import Config



class DataSynthesizer:
    def __init__(self, config: 'Config', existing_data_dir: Optional[str] = None):
        """
        Set up the data save directory; a new project directory (with ``args.yaml``) unless one is given.

        Args:
            config (Config): Data synthesis config.
            existing_data_dir (Optional[str], optional): Existing directory to save data into instead of a new project directory. Defaults to None.
        """
        # Initialize configuration, path and save directory
        self.config = config
        self._n = self.config.hospital_data.hospital_n
        if existing_data_dir is None:
            self.save_dir = make_project_dir(self.config)
            self.data_save_dir = self.save_dir / 'data'
            yaml_save(self.save_dir / 'args.yaml', self.config)
            os.makedirs(self.data_save_dir, exist_ok=True)
        else:
            self.data_save_dir = existing_data_dir
            os.makedirs(self.data_save_dir, exist_ok=True)
        getcontext().prec = 10
        

    @staticmethod
    def define_hospital_info(config: 'Config',
                             hospital_name: str,
                             department_info_path: Optional[str] = None,
                             with_follow_up: bool = False) -> Information:
        """
        Define the synthetic hospital data, including its departments and doctors.

        Args:
            config (Config): Data synthesis config (``hospital_data`` settings).
            hospital_name (str): Name of the hospital.
            department_info_path (Optional[str], optional): Department info file; the packaged ``department.json`` if None. Defaults to None.
            with_follow_up (bool, optional): If True, doctor schedules extend beyond the simulation window by the follow-up extra days. Defaults to False.

        Returns:
            Information: Hospital ``metadata``, ``department``, ``doctor`` and an empty ``patient``.

        Raises:
            AssertionError: If the follow-up extra days are negative or the department/doctor counts are inconsistent.
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

        # If follow_up_visit simulation included
        if with_follow_up:
            before_day = config.hospital_data.follow_up_visit.doctor_schedule_extra_days_before
            after_day = config.hospital_data.follow_up_visit.doctor_schedule_extra_days_after
            if before_day < 0:
                raise AssertionError(colorstr('red', f'`doctor_schedule_extra_days_before` ({before_day}) must be >= 0.'))
            if after_day < 0:
                raise AssertionError(colorstr('red', f'`doctor_schedule_extra_days_after` ({after_day}) must be >= 0.'))
            doctor_start_date = datetime_to_str(str_to_datetime(dates[0]) - timedelta(days=before_day), "%Y-%m-%d")
            doctor_schedule_days = generate_date_range(
                doctor_start_date, int(days + before_day + after_day)
            )
        else:
            doctor_schedule_days = [d for d in dates]

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
            doctor_schedule_start_date=doctor_schedule_days[0],
            doctor_schedule_end_date=doctor_schedule_days[-1],
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
        departments = DataSynthesizer.department_list_generator(department_n, file_path=department_info_path)
        doctors = DataSynthesizer.name_list_generator(doctor_n, prefix='Dr. ')   # Doctor names are unique across all departments
        for department_data, doc_n in zip(departments, doctor_n_per_department):
            department, dep_code = department_data

            # Add department information
            department_info[department] = {'code': dep_code if dep_code else 'NA', 'doctor': []}
            
            # Add doctor information
            for _ in range(doc_n):
                doctor = doctors.pop()
                department_info[department]['doctor'].append(doctor)
                specialty, spe_code = generate_random_specialty(department, department_info_path)
                capacity_per_hour = random.choice(doctor_capacity_per_hour_list)
                working_days = random.randint(
                    round(config.hospital_data.working_days.min * (len(doctor_schedule_days) / len(dates))),
                    round(config.hospital_data.working_days.max * (len(doctor_schedule_days) / len(dates)))
                )
                working_dates = sorted(random.sample(doctor_schedule_days, working_days))
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
                # Generate doctor schedules according to the doctor_schedule_days
                for date in doctor_schedule_days:
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
            file_path (Optional[str], optional): Hospital name file to sample names from. Defaults to None.
        
        Returns:
            list[str]: Sampled names, or ``hospital_001``-style names without a file.
        """
        if file_path:
            if variables.HOSPITALS is None:
                variables.HOSPITALS = [word.capitalize() for word in txt_load(file_path).split('\n') if word.strip()]
            return [f"{random.choice(variables.HOSPITALS)}" for _ in range(hospital_n)]
        
        zfill_l = len(str(hospital_n))
        return [f"hospital_{padded_int(i, zfill_l)}" for i in range(hospital_n)]

    
    @staticmethod
    def department_list_generator(department_n: int,
                                  file_path: Optional[str] = None) -> list[tuple[str, Optional[str]]]:
        """
        Generate a list of department names based on the number of departments.
        
        Args:
            department_n (int): Number of departments to generate.
            file_path (Optional[str], optional): Department JSON file; the packaged ``department.json`` if None. Defaults to None.
        
        Returns:
            list[tuple[str, Optional[str]]]: ``(department name, code)`` pairs.

        Raises:
            ValueError: If more departments are requested than the file provides.
        """
        if file_path == None:
            file_path = str(resources.files("h_adminsim.assets.departments").joinpath("department.json"))

        if file_path:
            if variables.DEPARTMENTS is None:
                specialty = json_load(file_path)['specialty']
                variables.DEPARTMENTS = [(k2, v2['code']) for v1 in specialty.values() for k2, v2 in v1['subspecialty'].items()]
            
            if department_n > len(variables.DEPARTMENTS):
                raise ValueError(f"Requested {department_n} departments, but only {len(variables.DEPARTMENTS)} available in {file_path}.")
        
            return random.sample(variables.DEPARTMENTS, department_n)
            
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
            n (int): Number of names to generate.
            first_name_file_path (Optional[str], optional): Path to a file containing first names. Defaults to None.
            last_name_file_path (Optional[str], optional): Path to a file containing last names. Defaults to None.
            prefix (Optional[str], optional): Prefix added to each name (e.g. ``'Dr. '``). Defaults to None.
            reject_list (Optional[list[str]], optional): Names to exclude. Defaults to None.
        
        Returns:
            list[str]: Shuffled list of names.
        """
        if first_name_file_path == None:
            first_name_file_path = str(resources.files("h_adminsim.assets.names").joinpath("firstname.txt"))
        if last_name_file_path == None:
            last_name_file_path = str(resources.files("h_adminsim.assets.names").joinpath("lastname.txt"))

        if prefix != None:
            assert isinstance(prefix, str), colorstr("red", "`prefix` must be a string type")
            names = [f'{prefix}{name}' for name in generate_random_names(n, first_name_file_path, last_name_file_path, reject_list)]
        else:
            names = [name for name in generate_random_names(n, first_name_file_path, last_name_file_path, reject_list)]
        random.shuffle(names)
        return names


    @staticmethod
    def second_preference_generator(primary_preference: str, preference_candidates: list[str]) -> list[str]:
        """
        Generate a list of preferences based on the initial primary_preference.

        Args:
            primary_preference (str): First priority of preference.
            preference_candidates (list[str]): List of available preference options.

        Returns:
            list[str]: List of preferences including first and second priority.
        """
        preference_list = [primary_preference]
        other_candidates = [p for p in preference_candidates if p != primary_preference]

        if other_candidates:
            second_preference = random.choice(other_candidates)
            preference_list.append(second_preference)
        
        return preference_list
