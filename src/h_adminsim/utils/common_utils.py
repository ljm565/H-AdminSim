import math
import time
import pytz
import random
from sconf import Config
from collections import defaultdict
from collections.abc import (
    Mapping, 
    MutableMapping, 
    MutableSequence,
)
from decimal import Decimal, getcontext
from datetime import datetime, timedelta
from google.genai.errors import ServerError, ClientError
from openai import (
    RateLimitError,
    BadRequestError,
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
)
from typing import Optional, Union, Any, Callable, Iterable

from h_adminsim.registry import variables
from h_adminsim.registry.d_class import ConversationState
from h_adminsim.registry.phrases import OPFU_UNAVAILABLE_PHRASE_PATIENT, OPFU_UNAVAILABLE_PHRASE_STAFF
from h_adminsim.registry.variables import HALF_DAY_BOUNDARY_HOUR, HALF_DAY_ALIASES, HALF_DAY_WORD, HALF_DAY_RANGE
from h_adminsim.utils import Information, log, colorstr



def exponential_backoff(retry_count: int,
                        base_delay: int = 5,
                        max_delay: int = 65,
                        jitter: bool = True) -> float:
    """
    Delay before the next API retry: ``min(base_delay * 2**retry_count, max_delay)``, optionally jittered by ±20%.

    Args:
        retry_count (int): Number of retries so far.
        base_delay (int, optional): Base delay in seconds. Defaults to 5.
        max_delay (int, optional): Maximum delay in seconds (before jitter). Defaults to 65.
        jitter (bool, optional): Whether to apply random jitter. Defaults to True.

    Returns:
        float: Delay in seconds.
    """
    delay = min(base_delay * (2 ** retry_count), max_delay)
    if jitter:
        delay = random.uniform(delay * 0.8, delay * 1.2)
    return delay



def padded_int(n: int, total_digit_l: int = 3) -> str:
    """
    Convert a non-negative integer to a zero-padded string.

    Args:
        n (int): The integer to convert.
        total_digit_l (int, optional): Minimum length of the output string. Defaults to 3.

    Returns:
        str: The zero-padded string.

    Raises:
        ValueError: If ``n`` is negative or ``total_digit_l`` is not positive.
    """
    if n < 0:
        raise ValueError(colorstr("red", "Negative integers are not supported"))
    if total_digit_l <= 0:
        raise ValueError(colorstr("red", "Total digit length must be a positive integer"))
    
    return str(n).zfill(total_digit_l)



def to_dict(obj: Any) -> Any:
    """
    Recursively convert ``Information`` objects (also inside dicts and lists) to plain dicts.

    Args:
        obj (Any): The object to convert.

    Returns:
        Any: The converted object; other values are returned as-is.
    """
    if isinstance(obj, Information):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(v) for v in obj]
    else:
        return obj



def str_to_datetime(iso_time: Union[str, datetime]) -> datetime:
    """
    Parse an ISO 8601 string to a datetime; other values are returned as-is.

    Args:
        iso_time (Union[str, datetime]): ISO 8601 string or datetime/date object.

    Returns:
        datetime: The parsed datetime, or the input if it was not a string.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        if isinstance(iso_time, str): 
            return datetime.fromisoformat(iso_time)
        return iso_time
    except:
        raise ValueError(colorstr("red", f"`iso_time` must be str or date format, but got {type(iso_time)}"))
    


def datetime_to_str(iso_time: Union[str, datetime], format: str) -> str:
    """
    Format a datetime as a string; strings are returned as-is.

    Args:
        iso_time (Union[str, datetime]): Datetime/date object or string.
        format (str): ``strftime`` format, e.g. ``"%Y-%m-%d"``.

    Returns:
        str: The formatted string, or the input string unchanged.

    Raises:
        ValueError: If ``iso_time`` has no ``strftime``.
    """
    try:
        if not isinstance(iso_time, str): 
            return iso_time.strftime(format)
        return iso_time
    except:
        raise ValueError(colorstr("red", f"`iso_time` must be str or date format, but got {type(iso_time)}"))



def generate_date_range(start_date: Union[str, datetime], 
                        days: int) -> list[str]:
    """
    Generate a list of dates starting from `start_date` for `days` days.

    Args:
        start_date (Union[str, datetime]): Start date in ISO format (YYYY-MM-DD).
        days (int): Number of days to include (including start_date).

    Returns:
        list[str]: List of date strings in ISO format.

    Raises:
        ValueError: If ``days`` is not positive.
    """
    if days <= 0:
        raise ValueError(colorstr("red", f"`days` must be larger than 0, but got {days}"))
    start_date = str_to_datetime(start_date)
    return [datetime_to_str(start_date + timedelta(days=i), "%Y-%m-%d") for i in range(days)]



def get_utc_offset(country_code: Optional[str] = None, 
                   time_zone: Optional[str] = None) -> str:
    """
    Current UTC offset (e.g. "+09:00") for a country code or time zone.
    One of the two is required; give ``time_zone`` for countries with several time zones.

    Args:
        country_code (Optional[str], optional): ISO 3166-1 alpha-2 country code (e.g., 'KR'). Defaults to None.
        time_zone (Optional[str], optional): IANA time zone (e.g., 'Asia/Seoul'). Defaults to None.

    Returns:
        str: The UTC offset of the specified time zone in the format "+HH:MM" or "-HH:MM".
    """
    assert not (country_code == None and time_zone == None), colorstr("red", "Either `country_code` or `time_zone` must be provided.")
    
    if variables.COUNTRY_TIMEZONE_MAP is None:
        variables.COUNTRY_TIMEZONE_MAP = dict(pytz.country_timezones)
    
    if country_code:
        time_zones = variables.COUNTRY_TIMEZONE_MAP.get(country_code.upper())
        time_zone = time_zone if len(time_zones) > 1 else time_zones[0]   # If the country has mulitple time zone, you should use `time_zone` argument
    
    time_zone = pytz.timezone(time_zone)
    offset = datetime.now(time_zone).utcoffset()

    # Convert offset to "+HH:MM" or "-HH:MM"
    hours, remainder = divmod(offset.total_seconds(), 3600)
    minutes = abs(remainder) // 60
    return f'{int(hours):+03d}:{int(minutes):02d}'



def hour_to_hhmmss(hours: Union[int, float]) -> str:
    """
    Convert decimal hours to "HH:MM:SS".

    Args:
        hours (Union[int, float]): Number of hours.

    Returns:
        str: The time as "HH:MM:SS".
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
                 date: Optional[Union[str, datetime]] = None,
                 utc_offset: Optional[str] = None) -> str:
    """
    Construct an ISO 8601 time string from a given hour, optional date, and optional UTC offset.

    Args:
        time_hour (Union[int, float]): Time expressed in hours (e.g., 9.5 → 09:30:00).
        date (Optional[Union[str, datetime]], optional): Date as 'YYYY-MM-DD' or a datetime; None means today. Defaults to None.
        utc_offset (Optional[str], optional): UTC offset as '+HH:MM' or '-HH:MM'; None means no offset. Defaults to None.

    Returns:
        str: ISO 8601 formatted datetime string.

    Raises:
        ValueError: If the `date` format is invalid.
    """
    if date == None:
        date = datetime_to_str(datetime.today(), '%Y-%m-%d')
    else:
        try:
            date = datetime_to_str(date, '%Y-%m-%d')
            datetime.strptime(date, '%Y-%m-%d')
        except ValueError:
            raise ValueError(colorstr("red", f"Invalid date format: '{date}'. Expected format is 'YYYY-MM-DD'."))

    time = hour_to_hhmmss(time_hour)

    if utc_offset:
        return f'{date}T{time}{utc_offset}'
    return f'{date}T{time}'



def iso_to_hour(iso_time: Union[str, datetime]) -> float:
    """
    Time of day of an ISO 8601 time, in float hours.

    Args:
        iso_time (Union[str, datetime]): ISO 8601 time string (e.g., '2025-07-17T09:30:00+09:00') or datetime.

    Returns:
        float: Time represented in float hours (e.g., 9.5)
    """
    dt = str_to_datetime(iso_time)
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    return hour + minute / 60 + second / 3600



def iso_to_date(iso_time: Union[str, datetime]) -> str:
    """
    Extract date information from an ISO 8601 time.

    Args:
        iso_time (Union[str, datetime]): ISO 8601 time string (e.g., '2025-07-17T09:30:00+09:00')

    Returns:
        str: Date represented in string (e.g. 2024-05-23)
    """
    iso_time = str_to_datetime(iso_time)
    return str(iso_time.date())



def generate_random_iso_time_between(min_iso_time: Union[str, datetime],
                                     max_iso_time: Union[str, datetime],
                                     epsilon: float = 1e-6) -> str:
    """
    Generate a random ISO 8601 time string strictly within (min_iso_time, max_iso_time).

    Args:
        min_iso_time (Union[str, datetime]): The lower bound ISO 8601 time string (exclusive).
        max_iso_time (Union[str, datetime]): The upper bound ISO 8601 time string (exclusive).
        epsilon (float, optional): Buffer in seconds to exclude both bounds. Defaults to 1e-6.

    Returns:
        str: A random ISO 8601 time string within the range.

    Raises:
        ValueError: If min_iso_time is not earlier than max_iso_time or epsilon is too large.
    """
    min_dt, max_dt = str_to_datetime(min_iso_time), str_to_datetime(max_iso_time)

    if not compare_iso_time(max_dt, min_dt):
        raise ValueError(colorstr("red", f"min_iso_time ({min_iso_time}) must be earlier than max_iso_time ({max_iso_time})"))

    total_seconds = (max_dt - min_dt).total_seconds()

    if total_seconds <= 2 * epsilon:
        raise ValueError(colorstr("red", "Time range is too small for the given epsilon to exclude both bounds."))

    # Exclude both bounds by starting from epsilon and ending at total_seconds - epsilon
    random_seconds = random.uniform(epsilon, total_seconds - epsilon)
    random_dt = min_dt + timedelta(seconds=random_seconds)

    return random_dt.isoformat(timespec='seconds')



def compare_iso_time(time1: Union[str, datetime], time2: Union[str, datetime]) -> bool:
    """
    Compare two times given in ISO 8601 format and determine if the first is later than the second.

    Args:
        time1 (Union[str, datetime]): The first time value as an ISO 8601 string or a datetime object.
        time2 (Union[str, datetime]): The second time value as an ISO 8601 string or a datetime object.

    Returns:
        bool: True if `time1` is later than `time2`, otherwise False.
    """
    return str_to_datetime(time1) > str_to_datetime(time2)



def hours_between(later: Union[str, datetime], earlier: Union[str, datetime]) -> float:
    """
    Hours from `earlier` to `later` (negative when `later` is actually earlier).

    Args:
        later (Union[str, datetime]): The later time as an ISO 8601 string or a datetime object.
        earlier (Union[str, datetime]): The earlier time as an ISO 8601 string or a datetime object.

    Returns:
        float: `later - earlier` in hours.
    """
    return (str_to_datetime(later) - str_to_datetime(earlier)).total_seconds() / 3600



def add_hours_to_iso(iso_time: Union[str, datetime], hours: Union[int, float]) -> str:
    """
    Add `hours` to an ISO 8601 time, keeping a string's timezone suffix and carrying overflow into the date.

    Args:
        iso_time (Union[str, datetime]): ISO 8601 time string (e.g. '2026-05-01T17:00:00+09:00') or a datetime.
        hours (Union[int, float]): Hours to add. Can exceed 24.

    Returns:
        str: ISO 8601 string with hours added (e.g. '2026-05-03T17:00:00+09:00').
    """
    if isinstance(iso_time, str):
        suffix = ''
        for marker in ('+', '-'):
            tz_idx = iso_time.rfind(marker)
            if tz_idx > 10:
                suffix = iso_time[tz_idx:]
                iso_time = iso_time[:tz_idx]
                break
        ready = str_to_datetime(iso_time) + timedelta(hours=float(hours))
        return ready.isoformat(timespec='seconds') + suffix
    return (iso_time + timedelta(hours=float(hours))).isoformat(timespec='seconds')



def generate_random_iso_date_between(min_date: Union[str, datetime],
                                     max_date: Union[str, datetime]) -> str:
    """
    Generate a random date between min_date and max_date (inclusive).

    Args:
        min_date (Union[str, datetime]): Minimum date in ISO format (YYYY-MM-DD) or a date object.
        max_date (Union[str, datetime]): Maximum date in ISO format (YYYY-MM-DD) or a date object.

    Returns:
        str: Random date in ISO format (YYYY-MM-DD).

    Raises:
        ValueError: If ``max_date`` is before ``min_date``.
    """
    min_date, max_date = str_to_datetime(min_date).date(), str_to_datetime(max_date).date()
    delta_days = (max_date - min_date).days
    if delta_days < 0:
        raise ValueError(colorstr("red", "max_date must be after or equal to min_date."))

    random_days = random.randint(0, delta_days)

    return datetime_to_str(min_date + timedelta(days=random_days), "%Y-%m-%d")



def convert_time_to_segment(start: float, 
                            end: float, 
                            interval: float, 
                            time_range: Optional[list[float]] = None) -> list[int]:
    """
    Indices of the ``interval``-sized segments of [start, end), e.g. (0, 24, 0.5) -> [0, ..., 47].

    Args:
        start (float): Start time in hours (e.g., 0.0 for 00:00).
        end (float): End time in hours (e.g., 24.0 for 24:00).
        interval (float): Segment length in hours (e.g., 0.5 for 30 minutes).
        time_range (Optional[list[float]], optional): ``[start_time, end_time]`` subrange to return segments for; None means all. Defaults to None.

    Returns:
        list[int]: Segment indices.
    """
    assert start < end, colorstr("red", "Start time must be less than end time")
    assert interval > 0, colorstr("red", "Interval must be greater than 0")
    
    getcontext().prec = 10
    num_segments = int((Decimal(str(end)) - Decimal(str(start))) / Decimal(str(interval)))

    if time_range == None:
        return list(range(num_segments))

    # Sanity check for time_range
    assert len(time_range) == 2, colorstr("red", "Time range must be composed of two float values")
    assert time_range[0] >= start and time_range[1] <= end, colorstr("red", "Time range must be within overall time bounds")
    assert time_range[0] < time_range[1], colorstr("red", "Start time of `time_range` must be less than its end time")
    
    start_idx = int((Decimal(str(time_range[0])) - Decimal(str(start))) / Decimal(str(interval)))
    end_idx = int((Decimal(str(time_range[1])) - Decimal(str(start))) / Decimal(str(interval)))

    return list(range(start_idx, end_idx))



def convert_segment_to_time(start: float, 
                            end: float, 
                            interval: float, 
                            segments: list[int]) -> tuple[float, float]:
    """
    Convert a run of consecutive segment indices back to its time span, e.g. (0, 24, 0.5, [0, 1, 2]) -> (0.0, 1.5).

    Args:
        start (float): Start time in hours (e.g., 0.0 for 00:00).
        end (float): End time in hours (e.g., 24.0 for 24:00).
        interval (float): Segment length in hours (e.g., 0.5 for 30 minutes).
        segments (list[int]): Consecutive segment indices (e.g., [0, 1, 2]).

    Returns:
        tuple[float, float]: ``(start, end)`` of the span in hours, with the end capped at ``end``.
    """
    assert start < end, colorstr("red", "Start time must be less than end time")
    assert interval > 0, colorstr("red", "Interval must be greater than 0")

    getcontext().prec = 10
    max_segments = int((Decimal(str(end)) - Decimal(str(start))) / Decimal(str(interval))) - 1
    
    # Sanity checking
    for s in segments:
        assert 0 <= s <= max_segments, colorstr("red", f"Segment index {s} out of range")

    if len(segments) > 1:
        for i in range(1, len(segments)):
            assert segments[i] == segments[i-1] + 1, colorstr("red", "Segment indices must be continuous (i.e., increasing by 1)")
    
    seg_start = Decimal(str(start)) + Decimal(str(segments[0] * interval))
    seg_end = min(Decimal(str(start)) + Decimal(str((segments[-1] + 1) * interval)), end)

    return float(seg_start), float(seg_end)



def group_consecutive_segments(segments: list[int]) -> list[list[int]]:
    """
    Split a list of integers into runs of consecutive values, e.g. [1, 2, 3, 5, 6, 8] -> [[1, 2, 3], [5, 6], [8]].
    The input order is kept, so pass a sorted list; it must not be empty.

    Args:
        segments (list[int]): Integer segments to group.

    Returns:
        list[list[int]]: Runs of consecutive integers.
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



def split_into_contiguous_chunks(segments: list[int], chunk_size: int) -> list[list[int]]:
    """
    Split segments into non-overlapping contiguous chunks of ``chunk_size``; shorter remainders are dropped.

    Args:
        segments (list[int]): Segment indices (any order).
        chunk_size (int): Required size of each chunk.

    Returns:
        list[list[int]]: Contiguous chunks, each of length ``chunk_size`` (empty if ``chunk_size <= 0``).
    """
    if not segments or chunk_size <= 0:
        return []

    consecutive_blocks = group_consecutive_segments(sorted(segments))

    chunks = []
    for block in consecutive_blocks:
        n_chunks = len(block) // chunk_size
        for i in range(n_chunks):
            chunks.append(block[i * chunk_size : (i + 1) * chunk_size])

    return chunks



def convert_time_list_to_merged_time(start: float,
                                     end: float,
                                     interval: float,
                                     time_list: list[Union[tuple[float, float], list[float]]]) -> list[Union[tuple[float, float], list[float]]]:
    """
    Merge time intervals that cover consecutive segments of the ``interval`` grid.

    Args:
        start (float): Start of the overall time range in hours (e.g., 9.0).
        end (float): End of the overall time range in hours (e.g., 17.0).
        interval (float): Segment length in hours (e.g., 0.5 for 30 minutes).
        time_list (list[Union[tuple[float, float], list[float]]]): ``(start_time, end_time)`` intervals to merge.

    Returns:
        list[Union[tuple[float, float], list[float]]]: Merged ``[start_time, end_time]`` lists (the input itself if empty).
    """
    if len(time_list) > 0:
        segments = sum([convert_time_to_segment(start, end, interval, t) for t in time_list], [])
        grouped = group_consecutive_segments(segments)
        time_list = [list(convert_segment_to_time(start, end, interval, group)) for group in grouped]
    return time_list



def sort_schedule(data: Union[dict, list]) -> Union[dict, list]:
    """
    Sort schedule data for deterministic ordering: lists are sorted, dicts are sorted by key with each value sorted.

    Args:
        data (Union[dict, list]): Schedule data.

    Returns:
        Union[dict, list]: Sorted schedule data.
    """
    if isinstance(data, list):
        return sorted(data)
    return {k: sorted(v) for k, v in dict(sorted(data.items())).items()}



def personal_id_to_birth_date(personal_id: str,
                              start_date: str = "1960-01-01",
                              end_date: str = "2000-12-31") -> str:
    """
    Convert a personal ID (YYMMDD) to a birth date (YYYY-MM-DD), choosing the century within the given year range.

    Args:
        personal_id (str): Personal ID in YYMMDD or YYMMDD-XXXX format.
        start_date (str, optional): Earliest possible birth date (YYYY-MM-DD). Defaults to "1960-01-01".
        end_date (str, optional): Latest possible birth date (YYYY-MM-DD). Defaults to "2000-12-31".

    Returns:
        str: Birth date in YYYY-MM-DD format (random century, unpadded, if no century fits).
    """
    yymmdd = personal_id.split("-")[0]

    yy = int(yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])

    start_year = datetime.fromisoformat(start_date).year
    end_year = datetime.fromisoformat(end_date).year

    # Century candidates (1900s, 2000s)
    candidates = []
    for century in (1900, 2000):
        year = century + yy
        try:
            date = datetime(year, mm, dd)
            if start_year <= year <= end_year:
                candidates.append(date)
        except ValueError:
            continue

    if not candidates:
        return f"{random.choice(['19', '20'])}{yy}-{mm}-{dd}"

    return candidates[0].strftime("%Y-%m-%d")



def init_result_dict() -> dict:
    """
    Empty result dictionary for a task run.

    Returns:
        dict: ``{'gt', 'pred', 'status', 'status_code', 'trial', 'dialog', 'token'}`` mapped to empty lists.
    """
    return {'gt': [], 'pred': [], 'status': [], 'status_code': [], 'trial': [], 'dialog': [], 'token': []}



def hour_to_str(x: Union[int, float]) -> str:
    """
    Compact hour rendering without trailing zeros: 9.0 -> '9', 9.25 -> '9.25'.

    Args:
        x (Union[int, float]): Hour value.

    Returns:
        str: The rendered hour.
    """
    return f"{round(float(x), 4):g}"



def free_windows(occupied: Iterable[Union[list, tuple, dict]], start_hour: float, end_hour: float) -> list[tuple[float, float]]:
    """
    Free (bookable) windows: the complement of the occupied intervals within [start_hour, end_hour].

    Args:
        occupied (Iterable[Union[list, tuple, dict]]): ``[start, end]`` pairs or ``{'start', 'end'}`` dicts, possibly overlapping/unsorted; a falsy value means none.
        start_hour (float): Opening hour.
        end_hour (float): Closing hour.

    Returns:
        list[tuple[float, float]]: Sorted ``(start, end)`` free windows (possibly empty).
    """
    ivs = []
    for it in (occupied or []):
        s, e = (it.get('start'), it.get('end')) if isinstance(it, dict) else (it[0], it[1])
        if s is None or e is None:
            continue
        s, e = float(s), float(e)
        if s < e:
            ivs.append((s, e))
    ivs.sort()

    free, cur = [], float(start_hour)
    for s, e in ivs:
        if s > cur:
            free.append((cur, min(s, float(end_hour))))
        cur = max(cur, e)
        if cur >= float(end_hour):
            break
    if cur < float(end_hour):
        free.append((cur, float(end_hour)))
    return [(a, b) for a, b in free if a < b]



def render_availability_tables(items: Iterable[tuple[str, dict]], start_hour: float, end_hour: float,
                               value_header: str = 'Free (bookable) windows, start-end') -> str:
    """
    Render availability as per-entity markdown tables of free (bookable) windows, for reasoning prompts.

    Args:
        items (Iterable[tuple[str, dict]]): ``(title, schedule)`` pairs, ``schedule`` being ``{date: occupied intervals}``.
        start_hour (float): Opening hour used to compute free windows.
        end_hour (float): Closing hour used to compute free windows.
        value_header (str, optional): Header of the free-windows column. Defaults to 'Free (bookable) windows, start-end'.

    Returns:
        str: Markdown tables (one `###` block per item), or a placeholder when `items` is empty.
    """
    out = []
    for title, schedule in items:
        out.append(f"### {title}")
        out.append(f"| Date | {value_header} |")
        out.append("|------|" + "-" * (len(value_header) + 2) + "|")
        for date in sorted(schedule or {}):
            fw = free_windows(schedule[date], start_hour, end_hour)
            cell = ", ".join(f"{hour_to_str(s)}-{hour_to_str(e)}" for s, e in fw) if fw else "— (fully booked)"
            out.append(f"| {date} | {cell} |")
        out.append("")
    return "\n".join(out).strip() or "(none)"



def preprocess_dialog(dialog: list[dict]) -> str:
    """
    Render a dialog as ``role: content`` lines with whitespace collapsed.

    Args:
        dialog (list[dict]): Turns with 'role' and 'content'.

    Returns:
        str: The rendered dialog.
    """
    return '\n'.join([f"{turn['role']}: {' '.join(turn['content'].split())}" for turn in dialog])



def preprocess_utterance(utterance: str) -> str:
    """
    Collapse whitespace in an utterance; if it contains "#ACCEPT", strip the tag and end it with punctuation.

    Args:
        utterance (str): The content of the utterance.

    Returns:
        str: The cleaned utterance ('Thanks.' if only the tag remained).
    """
    if "#ACCEPT" in utterance:
        utterance = utterance.replace("#ACCEPT", "")
        utterance = ' '.join(utterance.split())

        if not utterance:
            return 'Thanks.'
        
        if utterance[-1] not in "!.?":
            utterance += "."
        return utterance
    
    return ' '.join(utterance.split())



def run_with_retry(func: Callable, *args: Any, max_retries: int = 8, **kwargs: Any) -> Any:
    """
    Call ``func`` and retry server/bad-request API errors with exponential backoff.
    The last error is re-raised once ``max_retries`` retries are used up.

    Args:
        func (Callable): Function to call.
        *args (Any): Positional arguments for ``func``.
        max_retries (int, optional): Maximum number of retries. Defaults to 8.
        **kwargs (Any): Keyword arguments for ``func``.

    Returns:
        Any: The return value of ``func``.
    """
    retry_count = 0

    while 1:
        try:
            return func(*args, **kwargs)

        except (ServerError, InternalServerError, BadRequestError) as e:
            if retry_count >= max_retries:
                log(f"\nMax retries reached. Last error: {e}", level='error')
                raise e

            wait_time = exponential_backoff(retry_count)
            log(
                f"[{retry_count + 1}/{max_retries}] {type(e).__name__}: {e}. "
                f"Retrying in {wait_time:.1f} seconds...",
                level='warning',
            )
            time.sleep(wait_time)
            retry_count += 1



def is_fatal_api_error(e: Exception) -> bool:
    """
    Whether `e` is an LLM API infrastructure failure that should stop the run rather than count as a simulation failure.

    Args:
        e (Exception): The exception to classify.

    Returns:
        bool: True for quota / auth / connection / exhausted-retry server errors (OpenAI, vLLM, Gemini).
    """
    FATAL_API_ERRORS = (
        RateLimitError,         # OpenAI / vLLM 429, including exhausted quota
        AuthenticationError,
        PermissionDeniedError,
        APIConnectionError,     # Also covers APITimeoutError
        InternalServerError,    # Only reaches the caller after `run_with_retry` gave up
        ServerError,            # Same, for Gemini
    )
    GENAI_FATAL_CODES = {401, 403, 429}
    
    if isinstance(e, FATAL_API_ERRORS):
        return True
    return isinstance(e, ClientError) and getattr(e, 'code', None) in GENAI_FATAL_CODES



def staff_role(state: Optional[ConversationState] = None,
               role: Optional[str] = None) -> str:
    """
    Build the Staff log label, annotated with the MAS node that handled the turn.

    Args:
        state (Optional[ConversationState], optional): Conversation state containing the current agent. Defaults to None.
        role (Optional[str], optional): Fallback role name when ``state`` is not provided. Defaults to None.
    
    Returns:
        str: A colored ``Staff (<agent name>)`` label.
    """
    name = state.current_agent if state else role
    if name == 'orchestrator':
        name = 'orchest'
    elif name == 'first_visit_intake':
        name = 'opfv int'
    elif name == 'first_visit_scheduling':
        name = 'opfv sch'
    elif name == 'follow_up_visit_scheduling':
        name = 'opfu sch'
    
    label = f'({name})'
    return f"{colorstr('blue', 'Staff')} {colorstr('magenta', f'{label:<10}')}"



def normalize_unavailable(unavailable: dict) -> Optional[dict]:
    """
    Normalize the patient's test-time unavailability (data shape, or tool shape without `type`) to one canonical shape.
    Idempotent: normalizing an already-normalized record returns the same record.

    Args:
        unavailable (dict): Raw unavailability record.

    Returns:
        Optional[dict]: `{'type': 'day' | 'half_day', 'day': [YYYY-MM-DD, ...],
                        'half_day': None | 'am' | 'pm', 'explanation': str}`, or None when the
                        record blocks nothing (no type, no dates, or an unusable half-day value).
    """
    if not isinstance(unavailable, dict):
        return None

    dates = sorted({str(d) for d in (unavailable.get('day') or []) if d})
    if not dates:
        return None

    half_day = unavailable.get('half_day')
    half_day = HALF_DAY_ALIASES.get(str(half_day).strip().lower()) if half_day else None

    # `type` is authoritative when present (data shape); otherwise infer it from the half-day value (tool shape).
    unavailable_type = unavailable.get('type') or ('half_day' if half_day else 'day')
    if unavailable_type == 'day':
        half_day = None
    elif unavailable_type != 'half_day' or half_day is None:
        return None   # unknown type, or a half-day constraint without a usable half

    return {
        'type': unavailable_type,
        'day': dates,
        'half_day': half_day,
        'explanation': unavailable.get('explanation') or '',
    }



def unavailable_blocked_interval(half_day: str,
                                 start_hour: float,
                                 end_hour: float) -> Optional[list[float]]:
    """
    Operating-hours interval blocked by a half-day constraint (split at `HALF_DAY_BOUNDARY_HOUR`, clipped to operating hours).

    Args:
        half_day (str): `'am'` (morning blocked) or `'pm'` (afternoon blocked).
        start_hour (float): Hospital opening hour.
        end_hour (float): Hospital closing hour.

    Returns:
        Optional[list[float]]: `[blocked_start, blocked_end]` in hours, or None when the
                                constraint blocks nothing inside operating hours.
    """
    if half_day == 'am':
        blocked = [start_hour, min(HALF_DAY_BOUNDARY_HOUR, end_hour)]
    elif half_day == 'pm':
        blocked = [max(HALF_DAY_BOUNDARY_HOUR, start_hour), end_hour]
    else:
        return None
    return blocked if blocked[0] < blocked[1] else None



def describe_unavailable(unavailable: dict, audience: str = 'patient') -> str:
    """
    Render the patient's test-time unavailability as prompt-ready text.

    Args:
        unavailable (dict): Raw unavailability record (either shape accepted by `normalize_unavailable`).
        audience (str, optional): `'patient'` for second-person phrasing, `'staff'` for third-person. Defaults to 'patient'.

    Returns:
        str: One sentence describing what the patient cannot attend (or that nothing is blocked).
    """
    phrases = OPFU_UNAVAILABLE_PHRASE_PATIENT if audience == 'patient' \
        else OPFU_UNAVAILABLE_PHRASE_STAFF

    normalized = normalize_unavailable(unavailable)
    if normalized is None:
        return phrases['none']

    # Both the ISO date and its spoken form, so the patient can say it naturally and the staff can pass it back as ISO.
    rendered_dates = []
    for date in normalized['day']:
        try:
            spoken = str_to_datetime(f'{date}T00:00:00').strftime('%A, %B %d')
            rendered_dates.append(f'{date} ({spoken})')
        except Exception:
            rendered_dates.append(date)

    return phrases[normalized['type']].format(
        dates=', '.join(rendered_dates),
        half_day_word=HALF_DAY_WORD.get(normalized['half_day'], ''),
        half_day_range=HALF_DAY_RANGE.get(normalized['half_day'], ''),
        explanation=normalized['explanation'],
    )



def calculate_idle_wait(assignments: Iterable[dict], priority_floor: float = float('inf')) -> float:
    """
    Idle waiting hours the patient spends between consecutive same-day tests.

    Args:
        assignments (Iterable[dict]): Test assignments with `date`, `start`, `end`, `priority`; `start`/`end` are ISO strings or float hours (hour of day).
        priority_floor (float, optional): Only count gaps whose later test has a strictly smaller priority; inf counts every gap. Defaults to float('inf').

    Returns:
        float: Total (or priority-locked) idle waiting time in hours.
    """
    def _hour(v: Union[int, float, str, datetime]) -> float:
        return v if isinstance(v, (int, float)) else iso_to_hour(v)

    by_date = defaultdict(list)
    for a in assignments:
        by_date[a['date']].append(a)
    
    total = Decimal('0')
    for items in by_date.values():
        items.sort(key=lambda a: _hour(a['start']))
        for prev, cur in zip(items, items[1:]):
            if cur['priority'] >= priority_floor:
                continue
            gap = Decimal(str(_hour(cur['start']))) - Decimal(str(_hour(prev['end'])))
            if gap > 0:
                total += gap
    
    return float(total)



def replace_none_value(config: Any) -> Any:
    """
    Recursively replace the string `'None'` (how a reloaded `Config.dumps()` null reads back) with `None`.
    Mutable containers are updated in place so their types (`Config`, `Munch`) survive; tuples are rebuilt.

    Args:
        config (Any): A configuration object, container, or scalar value.

    Returns:
        Any: The same object with every `'None'` string replaced by `None`.
    """
    if isinstance(config, (Config, MutableMapping)):
        for key, value in config.items():
            config[key] = replace_none_value(value)
        return config
    if isinstance(config, MutableSequence):
        for idx, value in enumerate(config):
            config[idx] = replace_none_value(value)
        return config
    if isinstance(config, tuple):
        return tuple(replace_none_value(value) for value in config)
    return None if isinstance(config, str) and config == 'None' else config



def to_builtin_scalar(value: Any) -> Any:
    """
    Recursively cast scalars (e.g. `ruamel.yaml` `ScalarFloat`) to built-in types so `orjson` can serialize them.
    Mappings are rebuilt as `dict`; lists, tuples, and sets as `list`.

    Args:
        value (Any): Scalar or container to normalize; other objects are returned untouched.

    Returns:
        Any: The value with every scalar as a built-in `bool`/`int`/`float`/`str`.
    """
    if isinstance(value, bool):     # Must precede int: bool is an int subclass
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Mapping):
        return {to_builtin_scalar(k): to_builtin_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin_scalar(v) for v in value]
    return value



def encode_non_finite(value: Any) -> Any:
    """
    Recursively copy `value` with every infinite float written as the string `'inf'` / `'-inf'`.
    `orjson` would save `inf` as `null`, indistinguishable from `None`; apply it only to the copy being saved.

    Args:
        value (Any): Structure to save (dicts, lists, scalars).

    Returns:
        Any: A copy safe to save, with infinities as strings.
    """
    if isinstance(value, float) and math.isinf(value):
        return 'inf' if value > 0 else '-inf'
    if isinstance(value, dict):
        return {k: encode_non_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_non_finite(v) for v in value]
    return value
