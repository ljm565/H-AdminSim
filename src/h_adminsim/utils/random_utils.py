import uuid
import random
from datetime import datetime, timedelta
from typing import Tuple, Any, Union, Optional

from h_adminsim import registry
from h_adminsim.utils import log
from h_adminsim.utils.filesys_utils import txt_load, json_load
from h_adminsim.utils.common_utils import str_to_datetime, datetime_to_str



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



def generate_random_number_string(length: int) -> str:
    """
    Generate a random numeric string of the given length.

    Args:
        length (int): The desired length of the random numeric string.

    Returns:
        str: A string consisting of randomly generated digits (0-9) 
             with the specified length.
    """
    return ''.join(str(random.randint(0, 9)) for _ in range(length))



def generate_random_names(n: int,
                          first_name_file: str = 'asset/names/firstname.txt',
                          last_name_file: str = 'asset/names/lastname.txt') -> list[str]:
    """
    Generate a list of random names by combining first and last names from specified files.

    Args:
        n (int): Number of random names to generate.
        first_name_file (str, optional): Path to the file containing first names. Defaults to 'asset/names/firstname.txt'.
        last_name_file (str, optional): Path to the file containing last names. Defaults to 'asset/names/lastname.txt'.

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



def generate_random_address(address_file_path: str = 'asset/country/address.json',
                            format: str = '{street}, {district}, {city}') -> str:
    """
    Generate random address.

    Args:
        address_file (str, optional): Path to the file containing addresses. Defaults to 'asset/country/address.json'.
        format (str, optional): Format of the address.

    Returns:
        str: Random generated address based on the asset.
    """
    if registry.ADDRESSES is None:
        registry.ADDRESSES = json_load(address_file_path)
    
    kwargs = dict()
    if 'street' in format:
        kwargs['street'] = f'{generate_random_number_string(random.randint(1, 2))}, {random.choice(registry.ADDRESSES["street"])}'
    if 'neighborhood' in format:
        kwargs['neighborhood'] = random.choice(registry.ADDRESSES['neighborhood'])
    if 'district' in format:
        kwargs['district'] = random.choice(registry.ADDRESSES['district'])
    if 'county' in format:
        kwargs['county'] = random.choice(registry.ADDRESSES['county'])
    if 'city' in format:
        kwargs['city'] = random.choice(registry.ADDRESSES['city'])
    if 'state' in format:
        kwargs['state'] = random.choice(registry.ADDRESSES['state'])
    
    return format.format(**kwargs)



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
                            ensure_unique_department: bool = True,
                            min_n: int = 2,
                            max_n: int = 4,
                            verbose: bool = True) -> Union[dict, str]:
    """
    Generate a string of random symptom from pre-defined data file.

    Args:
        department (str, optional): A name of hospital department.
        symptom_file_path (str, optional): A path of pre-defined symptom data. Defaults to './asset/departments/symptom.json'.
        ensure_unique_department (bool): Ensure that the disease can only be treated in a single medical specialty.
        min_n (int, optional): Minimum number of symptoms to select. Defaults to 2.
        max_n (int, optional): Maximum number of symptoms to select. Defaults to 4.
        verbose (bool, optional): If True, print a warning message when no matching department is found. Defaults to True.

    Returns:
        Union[dict, str]:
            - dict: A randomly selected disease and its associated symptoms for the given department.  
            - str: The string `"{PLACEHOLDER}"` if the department is not found in the data.
    """
    if registry.SYMPTOM_MAP is None:
        registry.SYMPTOM_MAP = json_load(symptom_file_path)
    
    if department in registry.SYMPTOM_MAP:
        if ensure_unique_department:
            unique_diseases = [dis for dis in registry.SYMPTOM_MAP[department] if len(dis[list(dis.keys())[0]]['department']) == 1]
            if not len(unique_diseases):
                log(f"In the specified {department}, there is no disease that can be treated within that specialty.\
                      As a result, if the department prediction later turns out to be a different specialty,\
                      it may not align with the patient’s preferred primary physician’s department, which can cause errors in the scheduling simulation.", 'warning')
            disease_list = unique_diseases if len(unique_diseases) else registry.SYMPTOM_MAP[department]
            disease_info = random.choice(disease_list)
        else:
            disease_info = random.choice(registry.SYMPTOM_MAP[department])
        disease = list(disease_info.keys())[0]
        disease_info = {'disease': disease, **disease_info[disease]}
        symptom_n = min(random.randint(min_n, max_n), len(disease_info['symptom']))
        disease_info['symptom'] = random.sample(disease_info['symptom'], symptom_n)
        return disease_info
    
    if verbose:
        log(f'No matched department {department}. `{{PLACEHOLDER}}` string will return.', 'warning')
    return '{PLACEHOLDER}'



def generate_random_telecom(min_length: int = 8, 
                            max_length: int = 13,
                            country_code: str = 'KR',
                            country_to_dial_map_file: str = 'asset/country/country_code.json') -> str:
    """
    Generate a random telecom number including the country dialing code.

    Args:
        min_length (int, optional): The minimum length of the subscriber number (excluding country code). Defaults to 8.
        max_length (int, optional): The maximum length of the subscriber number (excluding country code). Defaults to 13.
        country_code (str, optional): The ISO country code to determine the dialing prefix. Default is 'KR' (South Korea).
        country_to_dial_map_file (str, optional): Path to the JSON file mapping country codes to their dialing prefixes.
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



def generate_random_date(start_date: Union[str, datetime] = '1960-01-01',
                         end_date: Union[str, datetime] = '2000-12-31') -> str:
    """
    Generate a random date string in 'YYYY-MM-DD' format between the given start and end dates.

    Args:
        start_date (Union[str, datetime], optional): The start date in 'YYYY-MM-DD' format. Default is '2000-01-01'.
        end_date (Union[str, datetime], optional): The end date in 'YYYY-MM-DD' format. Default is '2025-12-31'.

    Returns:
        str: A randomly generated date string in 'YYYY-MM-DD' format.
    """
    start = str_to_datetime(start_date)
    end = str_to_datetime(end_date)
    delta = (end - start).days
    random_days = random.randint(0, delta)
    random_date = start + timedelta(days=random_days)
    return datetime_to_str(random_date, '%Y-%m-%d')



def generate_random_id_number(start_date: Union[str, datetime] = '1960-01-01',
                              end_date: Union[str, datetime] = '2000-12-31',
                              birth_date: Optional[str] = None) -> str:
    """
    Generate a random ID number consisting of a birth date and a random numeric sequence.

    Args:
        start_date (Union[str, datetime], optional): The earliest possible date of birth 
                                                     to consider when generating a random date. Defaults to '1960-01-01'.
        end_date (Union[str, datetime], optional): The latest possible date of birth 
                                                   to consider when generating a random date. Defaults to '2000-12-31'.
        birth_date (Optional[str], optional): A specific birth date in 'YYYY-MM-DD' format. 
                                              If provided, this date is used instead of generating a random one. Defaults to None.

    Returns:
        str: A randomly generated ID number in the format 'YYMMDD-XXXXXXX', 
        where 'YYMMDD' is the birth date and 'XXXXXXX' is a 7-digit random number.
    """
    if not birth_date:
        birth_date = generate_random_date(start_date, end_date)
    birth_date = birth_date.replace('-', '')[2:]
    return f"{birth_date}-{generate_random_number_string(7)}"



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
                         If the department is not found, returns ('{PLACEHOLDER}', '{PLACEHOLDER}').
    """
    if registry.SPECIALTIES is None:
        department_data = json_load(specialty_path)['specialty']
        registry.SPECIALTIES = {k2: {'code': v2['code'], 'field': v2['field']} for v1 in department_data.values() for k2, v2 in v1['subspecialty'].items()}
    
    if department in registry.SPECIALTIES:
        index = random.choice(range(len(registry.SPECIALTIES[department]['field'])))
        return registry.SPECIALTIES[department]['field'][index], f"{registry.SPECIALTIES[department]['code']}-{index}"
    
    if verbose:
        log(f'No matched department {department}. `{{PLACEHOLDER}}` string will return.', 'warning')
    return '{PLACEHOLDER}', '{PLACEHOLDER}'
