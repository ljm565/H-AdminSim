import random
from typing import Tuple, Optional

from h_adminsim.utils.common_utils import (
    convert_segment_to_time,
    convert_time_to_segment, 
    group_consecutive_segments,
)



class ScheduleAssigner:
    def __init__(self, start: float, end: float, interval: float):
        """
        Initialize a ScheduleAssigner for generating random schedules or appointments.

        This class divides a given time range into fixed-size segments and provides methods
        to assign schedules or appointments by selecting and grouping these segments.

        Args:
            start (float): Start time in hours (e.g., 9.0 for 09:00).
            end (float): End time in hours (e.g., 18.0 for 18:00).
            interval (float): Time interval in hours for each segment (e.g., 0.5 for 30 minutes).
        """
        self.start = start
        self.end = end
        self.interval = interval
        self.segments = convert_time_to_segment(self.start, self.end, self.interval)


    def schedule_segment_assign(self,
                                p: float,
                                segments: Optional[list[int]] = None) -> list[list[int]]:
        """
        Randomly assign a proportion of schedule time segments into grouped consecutive blocks.

        This method selects a random subset of segments, where the number of segments is 
        determined by the proportion `p` (e.g., 0.5 means 50% of all segments).
        The selected segments are then grouped into lists of consecutive segment indices.

        Args:
            p (float): Proportion of total segments to assign, between 0 and 1.
            segments (Optional[list[int]], optional): Specific segments. Defaults to None.

        Returns:
            list[list[int]]: A list of groups, where each group is a list of consecutive segment indices.
                            For example, [[0, 1], [3, 4, 5], [7]].

        Example:
            If segments = [0, 1, 2, ..., 11] and p = 0.5,
            this function might return something like:
                [[0, 1], [4], [6, 7, 8]]

        Notes:
            - The segment indices are selected randomly each time the function is called.
            - Groups are always composed of consecutive indices from the selected subset.
        """
        segments = self.segments if segments == None else segments
        segment_n = round(len(segments) * p)

        if segment_n > 0:
            # Select random segments
            chosen_segments = random.sample(segments, segment_n)
            chosen_segments.sort()

            # Grouping consecutive segments
            grouped = group_consecutive_segments(chosen_segments)
            return grouped
        return []
    

    def block_segment_assign(self,
                             p: float,
                             block_size: int,
                             segments: Optional[list[int]] = None) -> list[list[int]]:
        """
        Randomly place non-overlapping fixed-size blocks until about a proportion `p` of segments is covered.

        Unlike `schedule_segment_assign`, which scatters single segments, each occupancy here is one
        `block_size`-long booking (e.g. one exam), so the free time left between blocks is mostly long
        enough to hold another booking of the same size.

        Args:
            p (float): Target proportion of segments to cover, between 0 and 1.
            block_size (int): Number of consecutive segments per block.
            segments (Optional[list[int]], optional): Specific segments. Defaults to None.

        Returns:
            list[list[int]]: Covered segments grouped into consecutive runs (adjacent blocks merge).
        """
        segments = sorted(self.segments if segments == None else segments)
        if block_size <= 0 or block_size > len(segments):
            return []

        # Stochastic rounding keeps the expected coverage at `p` even when it is not a whole number of blocks
        target = len(segments) * p / block_size
        block_n = int(target) + (random.random() < target - int(target))
        if block_n <= 0:
            return []

        segment_set = set(segments)
        starts = [s for s in segments if all(s + i in segment_set for i in range(block_size))]
        random.shuffle(starts)

        used = set()
        placed = 0
        for s in starts:
            block = set(range(s, s + block_size))
            if used.isdisjoint(block):
                used |= block
                placed += 1
                if placed == block_n:
                    break

        return group_consecutive_segments(sorted(used))


    def appointment_segment_assign(self,
                                   p: float,
                                   min_chunk_size: int,
                                   max_chunk_size: int,
                                   segments: Optional[list[int]] = None) -> list[list[int]]:
        """
        Randomly assign appointment time segments from the remaining (unassigned) segments.

        Args:
            p (float): Proportion of remaining segments to sample and assign.
            min_chunk_size (int): The minimum time segment size for each appointment.
            max_chunk_size (int): The maximum time segment size for each appointment.
            segments (Optional[list[int]], optional): Specific segments. Defaults to None.

        Returns:
            list[list[int]]: Newly assigned segments from the remaining pool, grouped consecutively.
        """
        segments = self.segments if segments == None else segments
        avg_chunk_size = (min_chunk_size + max_chunk_size) / 2
        segment_n = int(len(segments) * p // avg_chunk_size)

        if segment_n > 0:
            chosen = []
            used = set()
            max_index = max(segments)
            random.shuffle(segments)
            
            for s in segments:
                max_possible_size = min(max_chunk_size, max_index - s + 1)
                if max_possible_size < min_chunk_size:
                    continue

                chunk_size = random.randint(min_chunk_size, max_chunk_size)
                trip = set(range(s, s + chunk_size))

                if used.isdisjoint(trip):
                    chosen.append((s, chunk_size))
                    used |= trip
                    if len(chosen) == segment_n:
                        break

            return [list(range(start, start + size)) for start, size in sorted(chosen)]
        return []
    

    def __call__(self,
                 p: float,
                 is_appointment: bool = False,
                 segments: Optional[list[list[int]]] = None,
                 block_size: Optional[int] = None,
                 **kwargs) -> Tuple[list[list[int]], list[list[float]]]:
        """
        Generate grouped time ranges by randomly selecting and grouping a proportion of time segments.

        This method allows the ScheduleAssigner instance to be called directly with a proportion `p`.
        It selects a subset of time segments based on `p`, groups them into consecutive segment blocks,
        and converts each group of segment indices into their corresponding time values.

        Args:
            p (float): Proportion of total segments to select, between 0 and 1.
            is_appointment (bool, optional): Whether the generated schedules are for appointments. Defaults to False.
            segments (Optional[list[list[int]]], optional): Specific segemnts. Defaults to None.
            block_size (Optional[int], optional): If given (and not an appointment), place `block_size`-long
                                                  blocks instead of scattering single segments. Defaults to None.

        Returns:
            list[list[float]]: A list of grouped time segments. Each group is a list of time segemnt values.
                            For example: [[2, 3], [5, 6, 7]].
            list[list[float]]: A list of grouped time ranges. Each group is a list of time values (in hours),
                            corresponding to consecutive time segments.
                            For example: [[0.0, 0.5], [2.0, 3.0]].

        Example:
            >>> assigner = ScheduleAssigner(0, 12, 0.5)
            >>> assigner(0.25)
            [[1.0, 1.5], [4.5], [6.0, 6.5]]
        """
        if is_appointment:
            time_segments = self.appointment_segment_assign(p, segments=segments, **kwargs)
        elif block_size:
            time_segments = self.block_segment_assign(p, block_size, segments)
        else:
            time_segments = self.schedule_segment_assign(p, segments)

        if len(time_segments):
            return time_segments, [list(convert_segment_to_time(self.start, self.end, self.interval, segments)) for segments in  time_segments]
        return [], []
