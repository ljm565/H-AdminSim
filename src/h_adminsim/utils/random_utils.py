import uuid
import random
from importlib import resources
from typing import Any, Union, Optional
from datetime import datetime, timedelta

from h_adminsim.registry import variables
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.filesys_utils import txt_load, json_load
from h_adminsim.utils.common_utils import str_to_datetime, datetime_to_str



def random_uuid(is_develop: bool = False) -> str:
    """
    Generate a random UUID.

    Args:
        is_develop (bool, optional): Build it from `random` bits (reproducible under a seed) instead of ``uuid1``. Defaults to False.

    Returns:
        str: The generated UUID.
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
        str: Random digits (0-9) of the given length.
    """
    return ''.join(str(random.randint(0, 9)) for _ in range(length))



def generate_random_names(n: int,
                          first_name_file: Optional[str] = None,
                          last_name_file: Optional[str] = None,
                          reject_list: Optional[list[str]] = None) -> list[str]:
    """
    Generate unique random "First Last" names; duplicates get a numeric suffix.

    Args:
        n (int): Number of random names to generate.
        first_name_file (Optional[str], optional): Path to the file containing first names. Defaults to None.
        last_name_file (Optional[str], optional): Path to the file containing last names. Defaults to None.
        reject_list (Optional[list[str]], optional): Names to avoid. Defaults to None.

    Returns:
        list[str]: Sorted list of names.
    """
    reject_list = set(reject_list) if reject_list else set()
    if first_name_file is None:
        first_name_file = str(resources.files("h_adminsim.assets.names").joinpath("firstname.txt"))
    if last_name_file is None:
        last_name_file = str(resources.files("h_adminsim.assets.names").joinpath("lastname.txt"))

    if variables.FIRST_NAMES is None:
        variables.FIRST_NAMES = [word.capitalize() for word in txt_load(first_name_file).split('\n') if word.strip()]
    if variables.LAST_NAMES is None:
        variables.LAST_NAMES = [word.capitalize() for word in txt_load(last_name_file).split('\n') if word.strip()]

    # Ensure unique names
    names = set()
    duplicate_name_num = {rn: 1 for rn in set(reject_list)}
    while len(names) < n:
        first_name = random.choice(variables.FIRST_NAMES)
        last_name = random.choice(variables.LAST_NAMES)
        full_name = f'{first_name} {last_name}'

        if full_name in names or full_name in reject_list:
            duplicate_name_num[full_name] = duplicate_name_num.setdefault(full_name, 1) + 1
            full_name = f'{full_name}{duplicate_name_num[full_name]}'
                
        names.add(full_name)
    return sorted(list(names))



def generate_random_address(address_file_path: Optional[str] = None,
                            format: str = '{street}, {district}, {city}') -> str:
    """
    Generate a random address from the address asset.

    Args:
        address_file_path (Optional[str], optional): Path to the address JSON file; None uses the bundled asset. Defaults to None.
        format (str, optional): Address template with fields among street/neighborhood/district/county/city/state. Defaults to '{street}, {district}, {city}'.

    Returns:
        str: The generated address.
    """
    if address_file_path == None:
        address_file_path = str(resources.files("h_adminsim.assets.country").joinpath("address.json"))

    if variables.ADDRESSES is None:
        variables.ADDRESSES = json_load(address_file_path)
    
    kwargs = dict()
    if 'street' in format:
        kwargs['street'] = f'{generate_random_number_string(random.randint(1, 2))}, {random.choice(variables.ADDRESSES["street"])}'
    if 'neighborhood' in format:
        kwargs['neighborhood'] = random.choice(variables.ADDRESSES['neighborhood'])
    if 'district' in format:
        kwargs['district'] = random.choice(variables.ADDRESSES['district'])
    if 'county' in format:
        kwargs['county'] = random.choice(variables.ADDRESSES['county'])
    if 'city' in format:
        kwargs['city'] = random.choice(variables.ADDRESSES['city'])
    if 'state' in format:
        kwargs['state'] = random.choice(variables.ADDRESSES['state'])
    
    return format.format(**kwargs)



def generate_random_prob(has_schedule_prob: float, coverage_min: float, coverage_max: float) -> float:
    """
    Draw a doctor's schedule coverage ratio.

    Args:
        has_schedule_prob (float): Probability that a doctor has any schedule.
        coverage_min (float): Minimum proportion of total available hours the schedule can occupy.
        coverage_max (float): Maximum proportion of total available hours the schedule can occupy.

    Returns:
        float: 0.0 if the doctor has no schedule, else a value in [coverage_min, coverage_max].
    """
    has_schedule = random.random() < has_schedule_prob
    if not has_schedule:
        return 0.0

    return random.uniform(coverage_min, coverage_max)



def generate_random_symptom(department: str, 
                            symptom_file_path: Optional[str] = None,
                            ensure_unique_department: bool = True,
                            min_n: int = 2,
                            max_n: int = 4,
                            verbose: bool = True) -> Union[dict, str]:
    """
    Pick a random disease of a department with a random subset of its symptoms.

    Args:
        department (str): Hospital department name.
        symptom_file_path (Optional[str], optional): Path to the symptom JSON file; None uses the bundled asset. Defaults to None.
        ensure_unique_department (bool, optional): Prefer diseases treated by this department only. Defaults to True.
        min_n (int, optional): Minimum number of symptoms to select. Defaults to 2.
        max_n (int, optional): Maximum number of symptoms to select. Defaults to 4.
        verbose (bool, optional): Warn when the department is not found. Defaults to True.

    Returns:
        Union[dict, str]: ``{'disease', 'symptom', ...}``, or ``"{PLACEHOLDER}"`` if the department is unknown.
    """
    if symptom_file_path == None:
        symptom_file_path = str(resources.files("h_adminsim.assets.departments").joinpath("symptom.json"))

    if variables.SYMPTOM_MAP is None:
        variables.SYMPTOM_MAP = json_load(symptom_file_path)
    
    if department in variables.SYMPTOM_MAP:
        # Ensure that the disease can only be treated in a single medical specialty
        if ensure_unique_department:
            unique_diseases = [dis for dis in variables.SYMPTOM_MAP[department] if len(dis[list(dis.keys())[0]]['department']) == 1]
            if not len(unique_diseases):
                log(f"In the specified {department}, there is no disease that can be treated within that specialty.\
                      As a result, if the department prediction later turns out to be a different specialty,\
                      it may not align with the patient’s preferred primary physician’s department, which can cause errors in the scheduling simulation.", 'warning')
            disease_list = unique_diseases if len(unique_diseases) else variables.SYMPTOM_MAP[department]
            disease_info = random.choice(disease_list)
        else:
            disease_info = random.choice(variables.SYMPTOM_MAP[department])
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
                            country_to_dial_map_file: Optional[str] = None) -> str:
    """
    Generate a random telecom number including the country dialing code.

    Args:
        min_length (int, optional): The minimum length of the subscriber number (excluding country code). Defaults to 8.
        max_length (int, optional): The maximum length of the subscriber number (excluding country code). Defaults to 13.
        country_code (str, optional): ISO country code for the dialing prefix; unknown codes get no prefix. Defaults to 'KR'.
        country_to_dial_map_file (Optional[str], optional): Path to the country-to-dial-code JSON file. Defaults to None.

    Returns:
        str: The dialing code followed by random digits.
    """
    if country_to_dial_map_file == None:
        country_to_dial_map_file = str(resources.files("h_adminsim.assets.country").joinpath("country_code.json"))

    if variables.TELECOM_COUNTRY_CODE is None:
        variables.TELECOM_COUNTRY_CODE = json_load(country_to_dial_map_file)
    
    try:
        dial_code = variables.TELECOM_COUNTRY_CODE[country_code.upper()]
    except KeyError:
        dial_code = ''
    
    length = random.randint(min_length, max_length)

    return dial_code + ''.join(random.choices('0123456789', k=length))



def generate_random_date(start_date: Union[str, datetime] = '1960-01-01',
                         end_date: Union[str, datetime] = '2000-12-31') -> str:
    """
    Generate a random date string in 'YYYY-MM-DD' format between the given start and end dates.

    Args:
        start_date (Union[str, datetime], optional): The start date in 'YYYY-MM-DD' format. Defaults to '1960-01-01'.
        end_date (Union[str, datetime], optional): The end date in 'YYYY-MM-DD' format (inclusive). Defaults to '2000-12-31'.

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
        start_date (Union[str, datetime], optional): Earliest random birth date. Defaults to '1960-01-01'.
        end_date (Union[str, datetime], optional): Latest random birth date. Defaults to '2000-12-31'.
        birth_date (Optional[str], optional): Birth date ('YYYY-MM-DD') to use instead of a random one. Defaults to None.

    Returns:
        str: ID number 'YYMMDD-XXXXXXX' (birth date, then 7 random digits).
    """
    if not birth_date:
        birth_date = generate_random_date(start_date, end_date)
    birth_date = birth_date.replace('-', '')[2:]
    return f"{birth_date}-{generate_random_number_string(7)}"



def generate_random_code(category: str) -> str:
    """
    Random code for a category: 'use' (mobile/work), 'gender' (male/female), or 'distance' (near/medium/far).

    Args:
        category (str): One of 'use', 'gender', 'distance'.

    Returns:
        str: A randomly selected code.

    Raises:
        ValueError: If the category is not supported.
    """
    
    if category == 'use':
        return random.choice(['mobile', 'work'])
    elif category == 'gender':
        return random.choice(['male', 'female'])
    elif category == 'distance':
        return random.choice(['near', 'medium', 'far'])
    else:
        raise ValueError(colorstr("red", f"Unsupported category: {category}"))



def generate_random_code_with_prob(codes: list[Any],
                                   probs: list[float]) -> Any:
    """
    Pick one code at random with the given probabilities.

    Args:
        codes (list[Any]): Codes to choose from.
        probs (list[float]): Probability of each code; must sum to 1.

    Returns:
        Any: The chosen element of ``codes``.
    """
    assert round(sum(probs), 4) == 1, colorstr("red", f"The sum of the probabilities would be a 1, but got {probs}")
    assert len(codes) == len(probs), colorstr("red", f"The lengths of codes and probabilities must be the same, but got codes length: {len(codes)} and probs length: {len(probs)}")

    chosen = random.choices(population=codes, weights=probs, k=1)[0]

    return chosen



def generate_random_specialty(department: str,
                              specialty_path: Optional[str] = None, 
                              verbose: bool = True) -> tuple[str, str]:
    """
    Generate a random specialty and its corresponding code for a given department.

    Args:
        department (str): The name of the department to look up specialties for.
        specialty_path (Optional[str], optional): Path to the JSON file containing department-specialty mappings. Defaults to None.
        verbose (bool, optional): Warn when the department is not found. Defaults to True.

    Returns:
        tuple[str, str]: ``(specialty, code)`` with code ``'<department code>-<index>'``, or ``('{PLACEHOLDER}', '{PLACEHOLDER}')`` if unknown.
    """
    if specialty_path == None:
        specialty_path = str(resources.files("h_adminsim.assets.departments").joinpath("department.json"))

    if variables.SPECIALTIES is None:
        department_data = json_load(specialty_path)['specialty']
        variables.SPECIALTIES = {k2: {'code': v2['code'], 'field': v2['field']} for v1 in department_data.values() for k2, v2 in v1['subspecialty'].items()}
    
    if department in variables.SPECIALTIES:
        index = random.choice(range(len(variables.SPECIALTIES[department]['field'])))
        return variables.SPECIALTIES[department]['field'][index], f"{variables.SPECIALTIES[department]['code']}-{index}"
    
    if verbose:
        log(f'No matched department {department}. `{{PLACEHOLDER}}` string will return.', 'warning')
    return '{PLACEHOLDER}', '{PLACEHOLDER}'



def generate_random_occupation(occupation_file_path: Optional[str] = None) -> str:
    """
    Generate a random occupation from a predefined list of occupations. 

    Args:
        occupation_file_path (Optional[str], optional): Path to the occupation JSON file; None uses the bundled asset. Defaults to None.

    Returns:
        str: A randomly selected occupation.
    """
    if occupation_file_path == None:
        occupation_file_path = str(resources.files("h_adminsim.assets.patient").joinpath("occupation.json"))

    if variables.OCCUPATION is None:
        variables.OCCUPATION = json_load(occupation_file_path)
    
    return random.choice(list(variables.OCCUPATION))



def generate_random_test_preference(distance: str,
                                    occupation: str,
                                    preference_candidates: list[str],
                                    occupation_file_path: Optional[str] = None) -> str:
    """
    Draw a first-priority test scheduling preference from occupation data, shifted toward `visit_min` with distance.
    The shift weight per distance is `variables.DISTANCE_VISIT_MIN_WEIGHT` (near: occupation only).

    Args:
        distance (str): Distance from home to hospital (near, medium, or far).
        occupation (str): Occupation name to look up.
        preference_candidates (list[str]): Candidate preference types.
        occupation_file_path (Optional[str], optional): Path to the occupation JSON file; None uses the bundled asset. Defaults to None.

    Returns:
        str: First-priority test scheduling preference.

    Raises:
        ValueError: If the distance is not supported.
    """
    if distance not in variables.DISTANCE_VISIT_MIN_WEIGHT:
        raise ValueError(colorstr("red", f"Unsupported distance: {distance}"))

    if occupation_file_path == None:
        occupation_file_path = str(resources.files("h_adminsim.assets.patient").joinpath("occupation.json"))

    if variables.OCCUPATION is None:
        variables.OCCUPATION = json_load(occupation_file_path)

    occupation_pref = variables.OCCUPATION[occupation]['test_schedule_preference']['preference']
    preferred_type = occupation_pref['type']
    preferred_prob = occupation_pref['prob']

    # Occupation-based distribution over the preference candidates
    other_candidates = [p for p in preference_candidates if p != preferred_type]
    if preferred_type is None or preferred_type not in preference_candidates:
        # If the preferred type is not specified, uniform over the preference candidates
        occupation_probs = [1 / len(preference_candidates)] * len(preference_candidates)
    elif preferred_prob is None or not other_candidates:
        occupation_probs = [1.0 if p == preferred_type else 0.0 for p in preference_candidates]
    else:
        occupation_probs = [preferred_prob if p == preferred_type else (1 - preferred_prob) / len(other_candidates)
                            for p in preference_candidates]

    # Shift the distribution toward `visit_min` as the distance increases
    visit_min_weight = variables.DISTANCE_VISIT_MIN_WEIGHT[distance] if 'visit_min' in preference_candidates else 0.0
    probs = [(1 - visit_min_weight) * prob + (visit_min_weight if p == 'visit_min' else 0.0)
             for p, prob in zip(preference_candidates, occupation_probs)]

    return generate_random_code_with_prob(preference_candidates, probs)


def generate_random_occupation_unavailable(occupation: str,
                                           dates: list[str],
                                           occupation_file_path: Optional[str] = None) -> dict:
    """
    Generate unavailable schedule constraints based on occupation data.

    Args:
        occupation (str): Occupation name to look up.
        dates (list[str]): Simulation date range.
        occupation_file_path (Optional[str], optional): Path to the occupation JSON file; None uses the bundled asset. Defaults to None.

    Returns:
        dict: ``{'type', 'day', 'half_day', 'explanation'}``; ``type`` None when unconstrained.
    """
    if occupation_file_path == None:
        occupation_file_path = str(resources.files("h_adminsim.assets.patient").joinpath("occupation.json"))

    if variables.OCCUPATION is None:
        variables.OCCUPATION = json_load(occupation_file_path)

    unavailable_info = variables.OCCUPATION[occupation]['test_schedule_preference']['unavailable']
    unavailable_type = unavailable_info['type']
    unavailable_prob = unavailable_info['prob']
    unavailable_detail = unavailable_info['detail']
    explanation = unavailable_info['explanation']

    unavailable = {
        'type': None,
        'day': [],
        'half_day': None,
        'explanation': explanation if unavailable_type is None else '',
    }

    if unavailable_type is None or random.random() > unavailable_prob:
        return unavailable

    # Init detail conditions
    scale = len(dates) / 7
    unavailable_day_range = [d * scale for d in unavailable_detail['day']]
    unavailable_day_n = min(int(random.uniform(*unavailable_day_range)), len(dates))

    # A short date range can scale the day count down to zero; keep the record unconstrained in that case
    if unavailable_day_n <= 0:
        return unavailable

    unavailable['type'] = unavailable_type
    unavailable['day'] = sorted(random.sample(dates, unavailable_day_n))
    unavailable['explanation'] = explanation

    if unavailable_type == 'half_day':
        unavailable[unavailable_type] = random.choice(unavailable_detail[unavailable_type])

    return unavailable