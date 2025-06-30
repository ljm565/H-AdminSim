import os
import uuid
import random

import registry
from utils import log
from utils.filesys_utils import txt_load



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



def generate_random_names(n: int, first_name_file: str, last_name_file: str) -> list[str]:
    """
    Generate a list of random names by combining first and last names from specified files.

    Args:
        n (int): Number of random names to generate.
        first_name_file (str): Path to the file containing first names.
        last_name_file (str): Path to the file containing last names.

    Returns:
        list[str]: List of randomly generated names in the format "First Last".
    """
    if registry.FIRST_NAMES is None:
        registry.FIRST_NAMES = [word.capitalize() for word in txt_load(first_name_file).split('\n') if word.strip()]
    if registry.LAST_NAMES is None:
        registry.LAST_NAMES = [word.capitalize() for word in txt_load(last_name_file).split('\n') if word.strip()]

    # Ensure unique names
    names = set()
    while len(names) < n:
        first_name = random.choice(registry.FIRST_NAMES)
        last_name = random.choice(registry.LAST_NAMES)
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



def generate_random_symptom(department: str, asset_folder: str, verbose: bool = True) -> str:
    """
    Generate a string of random symptom from pre-defined data file.

    Args:
        department (str): A name of hospital department.
        asset_folder (str): A directory of pre-defined symptom data.
        verbose (bool): If True, print a warning message when no matching department is found. Defaults to True.

    Returns:
        str: A randomly selected symptom.
    """
    if not len(registry.SYMPTOM_MAP):
        for root, _, files in os.walk(asset_folder):
            for file in files:
                if file.endswith(".txt"):
                    dep = os.path.splitext(os.path.basename(file))[0].lower()
                    registry.SYMPTOM_MAP[dep] = [symptom for symptom in txt_load(os.path.join(root, file)).split('\n') if symptom.strip()]
                    
        print(registry.SYMPTOM_MAP)
    
    if department in registry.SYMPTOM_MAP:
        return random.choice(registry.SYMPTOM_MAP[department])
    
    if verbose:
        log(f'No matched department {department}. `${{PLACEHOLDER}}` string will return.', 'warning')
    return '${PLACEHOLDER}'
