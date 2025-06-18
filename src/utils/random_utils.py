import uuid
import random

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import log



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





if __name__ == '__main__':
    # Example usage
    print('----------- ')
    print(random_time_segment(0, 24, 0.5))
    print('----------- ')
    print(random_time_segment(0, 24, 1.0))
    print('----------- ')
    print(random_time_segment(0, 12, 0.25))
    print('----------- ')
    print(random_time_segment(9.6, 18, 0.5))
    print('----------- ')
    print(random_time_segment(0, 1, 0.1))