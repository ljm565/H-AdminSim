import uuid
import random

import registry
from utils import log
from utils.filesys_utils import txt_load



def random_uuid(is_develop:bool=False) -> str:
    """_summary_

    Args:
        is_develop (bool, optional): _description_. Defaults to False.

    Returns:
        str: _description_
    """
    if is_develop:
        # For development purposes, generate controlled random UUID
        rand_bytes = random.getrandbits(128).to_bytes(16, 'big')
        return str(uuid.UUID(bytes=rand_bytes))
    return str(uuid.uuid1())



def random_time_segment(start: float, end: float, interval: float) -> list[int]:
    """
    Generate time segment indices between a start and end time, using a specified interval.

    This function divides a time range (e.g., 0 to 24 hours) into equal segments 
    and returns a list of integer indices representing those segments.

    Args:
        start (float): Start time in hours (e.g., 0.0 for 00:00).
        end (float): End time in hours (e.g., 24.0 for 24:00).
        interval (float): Time interval in hours (e.g., 0.5 for 30 minutes).

    Returns:
        list[int]: List of segment indices, where each index corresponds to a time slot.

    Example:
        >>> generate_time_segments(0, 24, 0.5)
        [0, 1, 2, ..., 47]  # Represents 48 half-hour segments from 00:00 to 24:00
    """
    assert start < end, log("Start time must be less than end time", "error")
    assert interval > 0, log("Interval must be greater than 0", "error")

    num_segments = int((end - start) / interval)
    return list(range(num_segments))



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
        registry.FIRST_NAMES = [word.capitalize() for word in txt_load(first_name_file).split('\n')]
    if registry.LAST_NAMES is None:
        registry.LAST_NAMES = [word.capitalize() for word in txt_load(last_name_file).split('\n')]

    # Ensure unique names
    names = set()
    while len(names) < n:
        first_name = random.choice(registry.FIRST_NAMES)
        last_name = random.choice(registry.LAST_NAMES)
        names.add(f"{first_name} {last_name}")
    return sorted(list(names))
