import re
import pytz
import random
from decimal import Decimal, getcontext
from datetime import datetime, timedelta
from typing import Optional, Union, Tuple

import registry
from registry import Hospital
from utils import Information, log, colorstr



def exponential_backoff(retry_count: int,
                        base_delay: int = 5,
                        max_delay: int = 65,
                        jitter: bool = True) -> float:
    """
    Exponential backoff function for API calling.

    Args:
        retry_count (int): Retry count.
        base_delay (int, optional): Base delay seconds. Defaults to 5.
        max_delay (int, optional): Maximum delay seconds. Defaults to 165.
        jitter (bool, optional): Whether apply randomness. Defaults to True.

    Returns:
        float: Final delay time.
    """
    delay = min(base_delay * (2 ** retry_count), max_delay)
    if jitter:
        delay = random.uniform(delay * 0.8, delay * 1.2)
    return delay



def padded_int(n: int, total_digit_l: int = 3) -> str:
    """
    Convert an integer to a zero-padded string of length 2.

    Args:
        n (int): The integer to convert.
        total_digit_l (int): The total number of digits in the output string. Default is 3.

    Returns:
        str: The zero-padded string representation of the integer.
    """
    if n < 0:
        raise ValueError(colorstr("red", "Negative integers are not supported"))
    if total_digit_l <= 0:
        raise ValueError(colorstr("red", "Total digit length must be a positive integer"))
    
    return str(n).zfill(total_digit_l)



def to_dict(obj: Information) -> dict:
    """
    Convert an object to a dictionary representation.

    Args:
        obj (Information): The object to convert.

    Returns:
        dict: A dictionary representation of the object.
    """
    if isinstance(obj, Information):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(v) for v in obj]
    else:
        return obj



def convert_info_to_obj(data: Information) -> Hospital:
    """
    Convert an Information object to a Hospital object.

    Args:
        data (Information): The Information object to convert.

    Returns:
        Hospital: A Hospital object constructed from the Information data.
    """
    data = to_dict(data)  # Convert Information to dict for easier access
    
    # Make doctor to patient map due to weakly linked patient data in the data dictionary
    doctor_to_patient = dict()
    for patient, patient_values in data['patient'].items():
        doctor = patient_values['attending_physician']
        if doctor in doctor_to_patient:
            doctor_to_patient[doctor].append(patient)
        else:
            doctor_to_patient[doctor] = [patient]
    
    hospital_obj = Hospital(**data.get('metadata'))
    for department, department_values in data.get('department').items():
        filtered_values1 = {k: v for k, v in department_values.items() if k != 'doctor'}
        department_obj = hospital_obj.add_department(department, **filtered_values1)
        
        for doctor in department_values['doctor']:
            doctor_values = data.get('doctor').get(doctor)
            filtered_values2 = {k: v for k, v in doctor_values.items() if k != 'department'}
            doctor_obj = department_obj.add_doctor(doctor, **filtered_values2)
            for patient in doctor_to_patient[doctor]:
                patient_values = data.get('patient').get(patient)
                filtered_values3 = {k: v for k, v in patient_values.items() if k != 'attending_physician'}
                doctor_obj.add_patient(patient, **filtered_values3)
    
    return hospital_obj



def convert_obj_to_info(hospital_obj: Hospital) -> Information:
    """
    Convert a Hospital object to an Information object.

    Args:
        hospital (Hospital): The Hospital object to convert.

    Returns:
        Information: An Information object constructed from the Hospital data.
    """
    filtered_values = {k: v for k, v in hospital_obj.__dict__.items() if k not in ['department', 'time']}
    filtered_values['time'] = Information(**hospital_obj.time)
    metadata = Information(**filtered_values)
    
    department_info, doctor_info, patient_info = dict(), dict(), dict()
    for department_obj in hospital_obj.department:
        filtered_values = {k: v for k, v in department_obj.__dict__.items() if k not in ['name', 'doctor']}
        filtered_values['doctor'] = [doctor_obj.name for doctor_obj in department_obj.doctor]
        department_info[department_obj.name] = {**filtered_values}
        
        for doctor_obj in department_obj.doctor:
            filtered_values2 = {k: v for k, v in doctor_obj.__dict__.items() if k not in ['name', 'department', 'patient']}
            filtered_values2['department'] = doctor_obj.department.name
            doctor_info[doctor_obj.name] = {**filtered_values2}

            for patient_obj in doctor_obj.patient:
                filtered_values3 = {k: v for k, v in patient_obj.__dict__.items() if k not in ['name', 'department', 'attending_physician']}
                filtered_values3['department'] = doctor_obj.department.name
                filtered_values3['attending_physician'] = doctor_obj.name
                patient_info[patient_obj.name] = {**filtered_values3}

    return Information(metadata=metadata, department=department_info, doctor=doctor_info, patient=patient_info)



def generate_date_range(start_date: Union[str, datetime.date], 
                        days: int) -> list[str]:
    """
    Generate a list of dates starting from `start_date` for `days` days.

    Args:
        start_date (Union[str, date]): Start date in ISO format (YYYY-MM-DD).
        days (int): Number of days to include (including start_date).

    Returns:
        List[str]: List of date strings in ISO format.
    """
    if days <= 0:
        raise ValueError(colorstr("red", f"`days` must be larger than 0, but got {days}"))

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    return [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]



def get_utc_offset(country_code: Optional[str] = None, 
                   time_zone: Optional[str] = None) -> str:
    """
    Returns the current UTC offset (e.g., "+09:00") for a given country code or time zone.

    Either `country_code` or `time_zone` must be provided. If the country has multiple time zones,
    you should explicitly provide the `time_zone`.

    Args:
        country_code (Optional[str], optional): ISO 3166-1 alpha-2 country code (e.g., 'KR', 'US').
        time_zone (Optional[str], optional): IANA time zone string (e.g., 'Asia/Seoul', 'America/New_York').

    Returns:
        str: The UTC offset of the specified time zone in the format "+HH:MM" or "-HH:MM".
    """
    assert not (country_code == None and time_zone == None), log("Either `country_code` or `time_zone` must be provided.", "error")
    
    if registry.COUNTRY_TIMEZONE_MAP is None:
        registry.COUNTRY_TIMEZONE_MAP = dict(pytz.country_timezones)
    
    if country_code:
        time_zones = registry.COUNTRY_TIMEZONE_MAP.get(country_code.upper())
        time_zone = time_zone if len(time_zones) > 1 else time_zones[0]   # If the country has mulitple time zone, you should use `time_zone` argument
    
    time_zone = pytz.timezone(time_zone)
    offset = datetime.now(time_zone).utcoffset()

    # Convert offset to "+HH:MM" or "-HH:MM"
    hours, remainder = divmod(offset.total_seconds(), 3600)
    minutes = abs(remainder) // 60
    return f'{int(hours):+03d}:{int(minutes):02d}'



def hour_to_hhmmss(hours: Union[int, float]) -> str:
    """
    Converts a decimal number of hours into HH:MM:SS format.

    Args:
        hours (Union[int, float]): A float or integer representing the number of hours.

    Returns:
        A string in the format "HH:MM:SS".
    """
    # Create a timedelta object from the given hours
    td = timedelta(hours=hours)

    # Extract total seconds from the timedelta object
    total_seconds = int(td.total_seconds())

    # Calculate hours, minutes, and seconds
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60

    # Format the output with zero-padding
    return f'{h:02}:{m:02}:{s:02}'



def get_iso_time(time_hour: Union[int, float],
                 date: Optional[str] = None,
                 utc_offset: Optional[str] = None) -> str:
    """
    Construct an ISO 8601 time string from a given hour, optional date, and optional UTC offset.

    Args:
        time_hour (Union[int, float]): Time expressed in hours (e.g., 9.5 → 09:30:00).
        date (Optional[str], optional): Date string in 'YYYY-MM-DD' format. Defaults to today's date.
        utc_offset (Optional[str], optional): UTC offset in '+HH:MM' or '-HH:MM' format. Defaults to no offset.

    Returns:
        str: ISO 8601 formatted datetime string.

    Raises:
        ValueError: If the `date` format is invalid.
    """
    if date == None:
        date = datetime.today().strftime('%Y-%m-%d')
    else:
        try:
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(colorstr("red", f"Invalid date format: '{date}'. Expected format is 'YYYY-MM-DD'."))

    time = hour_to_hhmmss(time_hour)

    if utc_offset:
        return f'{date}T{time}{utc_offset}'
    return f'{date}T{time}'



def iso_to_hour(iso_time: str) -> float:
    """
    Extract time information from an ISO 8601 time and convert time to float hour.

    Args:
        iso_time_str (str): ISO 8601 time string (e.g., '2025-07-17T09:30:00+09:00')

    Returns:
        float: Time represented in float hours (e.g., 9.5)
    """
    iso_time_cleaned = re.sub(r'(\+|-)\d{2}:\d{2}$', '', iso_time)
    
    # Parse datetime (ignores timezone)
    dt = datetime.fromisoformat(iso_time_cleaned)

    hour = dt.hour
    minute = dt.minute
    second = dt.second

    return hour + minute / 60 + second / 3600



def iso_to_date(iso_time: str) -> str:
    """
    Extract date information from an ISO 8601 time.

    Args:
        iso_time (str): ISO 8601 time string (e.g., '2025-07-17T09:30:00+09:00')

    Returns:
        str: Date represented in string (e.g. 2024-05-23)
    """
    if isinstance(iso_time, str):
        iso_time = datetime.fromisoformat(iso_time)
    
    return str(iso_time.date())



def generate_random_iso_time_between(min_iso_time: str,
                                     max_iso_time: str,
                                     epsilon: float = 1e-6) -> str:
    """
    Generate a random ISO 8601 time string strictly within (min_iso_time, max_iso_time).

    Args:
        min_iso_time (str): The lower bound ISO 8601 time string (exclusive).
        max_iso_time (str): The upper bound ISO 8601 time string (exclusive).
        epsilon (float, optional): Small buffer to exclude both bounds. Defaults to 1e-6 seconds.

    Returns:
        str: A randomly generated ISO 8601 time string within the specified range.

    Raises:
        ValueError: If min_iso_time is not earlier than max_iso_time or epsilon is too large.
    """
    min_dt = datetime.fromisoformat(min_iso_time) if isinstance(min_iso_time, str) else min_iso_time
    max_dt = datetime.fromisoformat(max_iso_time) if isinstance(max_iso_time, str) else max_iso_time

    if not compare_iso_time(max_dt, min_dt):
        raise ValueError(colorstr("red", f"min_iso_time ({min_iso_time}) must be earlier than max_iso_time ({max_iso_time})"))

    total_seconds = (max_dt - min_dt).total_seconds()

    if total_seconds <= 2 * epsilon:
        raise ValueError(colorstr("red", "Time range is too small for the given epsilon to exclude both bounds."))

    # Exclude both bounds by starting from epsilon and ending at total_seconds - epsilon
    random_seconds = random.uniform(epsilon, total_seconds - epsilon)
    random_dt = min_dt + timedelta(seconds=random_seconds)

    return random_dt.isoformat()



def compare_iso_time(time1: str, time2: str) -> bool:
    """
    Compare two times given in ISO 8601 format and determine if the first is later than the second.

    Args:
        time1 (str): The first time value as an ISO 8601 string or a datetime object.
        time2 (str): The second time value as an ISO 8601 string or a datetime object.

    Returns:
        bool: True if `time1` is later than `time2`, otherwise False.
    """
    time1 = datetime.fromisoformat(time1) if isinstance(time1, str) else time1
    time2 = datetime.fromisoformat(time2) if isinstance(time2, str) else time2
    return time1 > time2



def generate_random_iso_date_between(min_date: Union[str, datetime.date],
                                     max_date: Union[str, datetime.date]) -> str:
    """
    Generate a random date between min_date and max_date (inclusive).

    Args:
        min_date Union[str, date]: Minimum date in ISO format (YYYY-MM-DD) or a date object.
        max_date Union[str, date]: Maximum date in ISO format (YYYY-MM-DD) or a date object.

    Returns:
        str: Random date in ISO format (YYYY-MM-DD).
    """
    if isinstance(min_date, str):
        min_date = datetime.strptime(min_date, "%Y-%m-%d").date()
    if isinstance(max_date, str):
        max_date = datetime.strptime(max_date, "%Y-%m-%d").date()

    delta_days = (max_date - min_date).days
    if delta_days < 0:
        raise ValueError(colorstr("red", "max_date must be after or equal to min_date."))

    random_days = random.randint(0, delta_days)

    return (min_date + timedelta(days=random_days)).strftime("%Y-%m-%d")



def convert_time_to_segment(start: float, 
                            end: float, 
                            interval: float, 
                            time_range: Optional[list[float]] = None) -> list[int]:
    """
    Generate time segment indices between a start and end time, using a specified interval.

    This function divides a time range (e.g., 0 to 24 hours) into equal segments 
    and returns a list of integer indices representing those segments.

    Args:
        start (float): Start time in hours (e.g., 0.0 for 00:00).
        end (float): End time in hours (e.g., 24.0 for 24:00).
        interval (float): Time interval in hours (e.g., 0.5 for 30 minutes).
        time_range (Optional[list[float]], optional): If provided, should be a list of two floats 
            [start_time, end_time]. Only segments within this subrange are returned.

    Returns:
        list[int]: List of segment indices, where each index corresponds to a time slot.

    Example:
        >>> generate_time_segments(0, 24, 0.5)
        [0, 1, 2, ..., 47]  # Represents 48 half-hour segments from 00:00 to 24:00
    """
    assert start < end, log("Start time must be less than end time", "error")
    assert interval > 0, log("Interval must be greater than 0", "error")
    
    getcontext().prec = 10
    num_segments = int((end - start) / interval)

    if time_range == None:
        return list(range(num_segments))

    # Sanity check for time_range
    assert len(time_range) == 2, log("Time range must be composed of two float values", "error")
    assert time_range[0] >= start and time_range[1] <= end, log("Time range must be within overall time bounds", "error")
    assert time_range[0] < time_range[1], log("Start time of `time_range` must be less than its end time", "error")
    
    start_idx = int((Decimal(str(time_range[0])) - Decimal(str(start))) / Decimal(str(interval)))
    end_idx = int((Decimal(str(time_range[1])) - Decimal(str(start))) / Decimal(str(interval)))

    return list(range(start_idx, end_idx))



def convert_segment_to_time(start: float, 
                            end: float, 
                            interval: float, 
                            segments: list[int]) -> Tuple[float, float]:
    """
    Convert segment indices back to actual time values based on the given start time and interval.

    Args:
        start (float): Start time in hours (e.g., 0.0 for 00:00).
        end (float): End time in hours (e.g., 24.0 for 24:00).
        interval (float): Time interval in hours (e.g., 0.5 for 30 minutes).
        segments (list[int]): List of segment indices to convert (e.g., [0, 1, 2]).

    Returns:
        list[float]: List of time values (in hours) corresponding to the given segments.

    Example:
        >>> convert_segment_to_time(0, 24, 0.5, [0, 1, 2])
        [0.0, 0.5, 1.0]
    """
    assert start < end, log("Start time must be less than end time", "error")
    assert interval > 0, log("Interval must be greater than 0", "error")

    getcontext().prec = 10
    max_segments = int((end - start) / interval) - 1
    
    # Sanity checking
    for s in segments:
        assert 0 <= s <= max_segments, log(f"Segment index {s} out of range", "error")

    if len(segments) > 1:
        for i in range(1, len(segments)):
            assert segments[i] == segments[i-1] + 1, log("Segment indices must be continuous (i.e., increasing by 1)", "error")
    
    seg_start = Decimal(str(start)) + Decimal(str(segments[0] * interval))
    seg_end = min(Decimal(str(start)) + Decimal(str((segments[-1] + 1) * interval)), end)

    return float(seg_start), float(seg_end)



def group_consecutive_segments(segments: list[int]) -> list[list[int]]:
    """
    Group a list of integer segments into consecutive blocks.

    This function takes a list of integers and splits it into sublists where 
    each sublist contains consecutive numbers. Numbers that are not consecutive
    start a new group.

    Args:
        segments (list[int]): A list of integer segments to be grouped. 
                              The list should contain integers in any order; 
                              typically sorted for meaningful consecutive grouping.

    Returns:
        list[list[int]]: A list of lists, where each sublist contains consecutive integers 
                         from the input list.
    
    Example:
        >>> group_consecutive_segments([1, 2, 3, 5, 6, 8])
        [[1, 2, 3], [5, 6], [8]]
    """
    consecutive_blocks = []
    group = [segments[0]]
    for i in range(1, len(segments)):
        if segments[i] == segments[i - 1] + 1:
            group.append(segments[i])
        else:
            consecutive_blocks.append(group)
            group = [segments[i]]
    consecutive_blocks.append(group)
    return consecutive_blocks



def convert_time_list_to_merged_time(start: float,
                                     end: float,
                                     interval: float,
                                     time_list: list[Tuple[float, float]]) -> list[Tuple[float, float]]:
    """
    Convert a list of time intervals into merged intervals based on a fixed segment grid.

    This function maps each time interval in `time_list` into discrete segments
    defined by the `start`, `end`, and `interval`. Consecutive segments are then
    merged back into continuous time intervals.

    Args:
        start (float): The start time of the overall time range (e.g., 9.0 for 9:00 AM).
        end (float): The end time of the overall time range (e.g., 17.0 for 5:00 PM).
        interval (float): The length of each time segment (e.g., 0.5 for 30 minutes).
        time_list (list[Tuple[float, float]]): A list of time intervals to convert and merge.
                                               Each interval is a tuple (start_time, end_time).

    Returns:
        list[Tuple[float, float]]: A list of merged time intervals as tuples (start_time, end_time).
                                   Intervals are merged if they cover consecutive segments.

    Example:
        >>> convert_time_list_to_merged_time(9.0, 17.0, 0.5, [(9.0, 10.0), (10.0, 11.0), (13.0, 14.0)])
        [(9.0, 11.0), (13.0, 14.0)]
    """
    if len(time_list) > 0:
        segments = sum([convert_time_to_segment(start, end, interval, t) for t in time_list], [])
        grouped = group_consecutive_segments(segments)
        time_list = [list(convert_segment_to_time(start, end, interval, group)) for group in grouped]
    return time_list
