import uuid
import random
from typing import Tuple, Any
from datetime import datetime, timedelta

import registry
from utils import log
from utils.filesys_utils import txt_load, json_load



def random_uuid(is_develop: bool = False) -> str:
    """
    Generate ranodm UUID

    Args:
        is_develop (bool, optional): If True, generates a UUID using fully random bytes
                                     to support reproducibility during development or debugging.
                                     Defaults to False.

    Returns:
        str: The generated UUID
    """
    if is_develop:
        # For development purposes, generate controlled random UUID
        rand_bytes = random.getrandbits(128).to_bytes(16, 'big')
        return str(uuid.UUID(bytes=rand_bytes))
    return str(uuid.uuid1())



def generate_random_names(n: int,
                          first_name_file: str = 'asset/names/firstname.txt',
                          last_name_file: str = 'asset/names/lastname.txt') -> list[str]:
    """
    Generate a list of random names by combining first and last names from specified files.

    Args:
        n (int): Number of random names to generate.
        first_name_file (str): Path to the file containing first names. Defaults to 'asset/names/firstname.txt'.
        last_name_file (str): Path to the file containing last names. Defaults to 'asset/names/lastname.txt'.

    Returns:
        list[str]: List of randomly generated names in the format "First Last".
    """
    if registry.FIRST_NAMES is None:
        registry.FIRST_NAMES = [word.capitalize() for word in txt_load(first_name_file).split('\n') if word.strip()]
    if registry.LAST_NAMES is None:
        registry.LAST_NAMES = [word.capitalize() for word in txt_load(last_name_file).split('\n') if word.strip()]

    # Ensure unique names
    duplicate_name_num, names = dict(), set()
    while len(names) < n:
        first_name = random.choice(registry.FIRST_NAMES)
        last_name = random.choice(registry.LAST_NAMES)
        full_name = f'{first_name} {last_name}'

        if full_name in names:
            duplicate_name_num[full_name] = duplicate_name_num.setdefault(full_name, 1) + 1
            full_name = f'{full_name}{duplicate_name_num[full_name]}'
                
        names.add(f"{first_name} {last_name}")
    return sorted(list(names))



def generate_random_prob(has_schedule_prob: float, coverage_min: float, coverage_max: float) -> float:
    """
    Determine the final schedule ratio for a doctor.

    Args:
        has_schedule_prob (float): Probability that a doctor has any schedule.
        coverage_min (float): Minimum proportion of total available hours the schedule can occupy.
        coverage_max (float): Maximum proportion of total available hours the schedule can occupy.

    Returns:
        float: The final schedule ratio. 0.0 if the doctor has no schedule. A float in [coverage_min, coverage_max] if the doctor has a schedule.
    """
    has_schedule = random.random() < has_schedule_prob
    if not has_schedule:
        return 0.0

    return random.uniform(coverage_min, coverage_max)



def generate_random_symptom(department: str, 
                            symptom_file_path: str = './asset/departments/symptom.json',
                            verbose: bool = True) -> str:
    """
    Generate a string of random symptom from pre-defined data file.

    Args:
        department (str): A name of hospital department.
        symptom_file_path (str): A path of pre-defined symptom data. Defaults to './asset/departments/symptom.json'.
        verbose (bool): If True, print a warning message when no matching department is found. Defaults to True.

    Returns:
        str: A randomly selected symptom.
    """
    if registry.SYMPTOM_MAP is None:
        registry.SYMPTOM_MAP = json_load(symptom_file_path)
    
    if department in registry.SYMPTOM_MAP:
        return random.choice(registry.SYMPTOM_MAP[department])
    
    if verbose:
        log(f'No matched department {department}. `${{PLACEHOLDER}}` string will return.', 'warning')
    return '${PLACEHOLDER}'



def generate_random_telecom(min_length: int = 8, 
                            max_length: int = 13,
                            country_code: str = 'KR',
                            country_to_dial_map_file: str = 'asset/country/country_code.json') -> str:
    """
    Generate a random telecom number including the country dialing code.

    Args:
        min_length (int): The minimum length of the subscriber number (excluding country code). Defaults to 8.
        max_length (int): The maximum length of the subscriber number (excluding country code). Defaults to 13.
        country_code (str): The ISO country code to determine the dialing prefix. Default is 'KR' (South Korea).
        country_to_dial_map_file (str): Path to the JSON file mapping country codes to their dialing prefixes.
                                        Defaults to 'asset/country/country_code.json'.

    Returns:
        str: A random telecom number string starting with the country dialing code, followed by a random sequence of digits.
        
    Raises:
        KeyError: If the provided country_code is not found in the dialing code registry.
    """
    if registry.TELECOM_COUNTRY_CODE is None:
        registry.TELECOM_COUNTRY_CODE = json_load(country_to_dial_map_file)
    
    try:
        dial_code = registry.TELECOM_COUNTRY_CODE[country_code.upper()]
    except KeyError:
        dial_code = ''
    
    length = random.randint(min_length, max_length)

    return dial_code + ''.join(random.choices('0123456789', k=length))



def generate_random_date(start_date: str = '1960-01-01', end_date: str = '2000-12-31') -> str:
    """
    Generate a random date string in 'YYYY-MM-DD' format between the given start and end dates.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format. Default is '2000-01-01'.
        end_date (str): The end date in 'YYYY-MM-DD' format. Default is '2025-12-31'.

    Returns:
        str: A randomly generated date string in 'YYYY-MM-DD' format.
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    delta = (end - start).days
    random_days = random.randint(0, delta)
    random_date = start + timedelta(days=random_days)
    return random_date.strftime('%Y-%m-%d')



def generate_random_code(category: str) -> str:
    """
    Generate a random code value based on the specified category.

    Supported categories:
        - 'use': returns either 'mobile' or 'work'
        - 'gender': returns either 'male' or 'female'

    Args:
        category (str): The category for which to generate a code. Must be one of ['use', 'gender'].

    Returns:
        str: A randomly selected code corresponding to the given category.

    Raises:
        AssertionError: If the category is not one of the supported values.
    """
    categories = ['use', 'gender']
    assert category in categories, log(f"The category must be one of the values in the {categories}, but got {category}", "error")
    
    if category == 'use':
        return random.choice(['mobile', 'work'])
    elif category == 'gender':
        return random.choice(['male', 'female'])



def generate_random_code_with_prob(codes: list[Any],
                                   probs: list[float]) -> Tuple[int, str]:
    """
    Select a random code from the given list of codes based on the provided probabilities.

    Args:
        codes (list[Any]): A list of codes from which to choose.
        probs (list[float]): A list of probabilities corresponding to each code. 
                             The probabilities must sum to 1.

    Returns:
        Tuple[int, str]: The randomly chosen code. The exact type depends on the elements of `codes`.
                         (Adjust this description if the return type is known specifically.)
    """
    assert round(sum(probs), 4) == 1, log(f"The sum of the probabilities would be a 1, but got {probs}", "error")
    assert len(codes) == len(probs), log(f"The lengths of codes and probabilities must be the same, but got codes length: {len(codes)} and probs length: {len(probs)}", "error")

    chosen = random.choices(population=codes, weights=probs, k=1)[0]

    return chosen



def generate_random_specialty(department: str, 
                              specialty_path: str = 'asset/departments/department.json', 
                              verbose: bool = True) -> Tuple[str, str]:
    """
    Generate a random specialty and its corresponding code for a given department.

    Args:
        department (str): The name of the department to look up specialties for.
        specialty_path (str, optional): Path to the JSON file containing department-specialty mappings.
                                        Defaults to 'asset/departments/department.json'.
        verbose (bool, optional): Whether to log a warning message if the department is not found.
                                  Defaults to True.

    Returns:
        Tuple[str, str]: A tuple containing a randomly selected specialty (str) and its code (int).
                         If the department is not found, returns ('${PLACEHOLDER}', '${PLACEHOLDER}').
    """
    if registry.SPECIALTIES is None:
        department_data = json_load(specialty_path)['specialty']
        registry.SPECIALTIES = {k2: {'code': v2['code'], 'field': v2['field']} for v1 in department_data.values() for k2, v2 in v1['subspecialty'].items()}
    
    if department in registry.SPECIALTIES:
        index = random.choice(range(len(registry.SPECIALTIES[department]['field'])))
        return registry.SPECIALTIES[department]['field'][index], f"{registry.SPECIALTIES[department]['code']}-{index}"
    
    if verbose:
        log(f'No matched department {department}. `${{PLACEHOLDER}}` string will return.', 'warning')
    return '${PLACEHOLDER}', '${PLACEHOLDER}'
