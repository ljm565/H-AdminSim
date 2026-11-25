import random
from typing import Optional

from h_adminsim.utils.common_utils import (
    convert_segment_to_time,
    convert_time_to_segment, 
    group_consecutive_segments,
)



class ScheduleAssigner:
    def __init__(self, start: float, end: float, interval: float):
        """
        Divide a time range into fixed-size segments for random schedule / appointment assignment.

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
        Randomly pick ``round(len(segments) * p)`` segments and group them into consecutive runs.

        Args:
            p (float): Proportion of total segments to assign, between 0 and 1.
            segments (Optional[list[int]], optional): Specific segments. Defaults to None.

        Returns:
            list[list[int]]: Groups of consecutive segment indices, e.g. [[0, 1], [3, 4, 5], [7]].
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

        Each block is one `block_size`-long booking (e.g. one exam), so the gaps left mostly fit another such booking.

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
                 segments: Optional[list[int]] = None,
                 block_size: Optional[int] = None,
                 **kwargs) -> tuple[list[list[int]], list[list[float]]]:
        """
        Randomly select a proportion `p` of time segments, group them into consecutive runs and convert them to times.

        Args:
            p (float): Proportion of total segments to select, between 0 and 1.
            is_appointment (bool, optional): Whether the generated schedules are for appointments. Defaults to False.
            segments (Optional[list[int]], optional): Specific segments (None -> all). Defaults to None.
            block_size (Optional[int], optional): If given (and not an appointment), place `block_size`-long
                                                  blocks instead of scattering single segments. Defaults to None.
            **kwargs (Any): ``min_chunk_size`` / ``max_chunk_size`` for `appointment_segment_assign`.

        Returns:
            tuple[list[list[int]], list[list[float]]]: The segment groups (e.g. [[2, 3], [5, 6, 7]]) and the
                                                       corresponding time values in hours (``([], [])`` if none).
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
