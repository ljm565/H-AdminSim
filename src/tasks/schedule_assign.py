import random
from typing import Tuple, Optional

from utils.common_utils import convert_segment_to_time, convert_time_to_segment



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
                                segments: Optional[list[list[int]]] = None) -> list[list[int]]:
        """
        Randomly assign a proportion of schedule time segments into grouped consecutive blocks.

        This method selects a random subset of segments, where the number of segments is 
        determined by the proportion `p` (e.g., 0.5 means 50% of all segments).
        The selected segments are then grouped into lists of consecutive segment indices.

        Args:
            p (float): Proportion of total segments to assign, between 0 and 1.
            segments (Optional[list[list[int]]], optional): Specific segemnts. Defaults to None.

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
            grouped = []
            group = [chosen_segments[0]]
            for i in range(1, len(chosen_segments)):
                if chosen_segments[i] == chosen_segments[i-1] + 1:
                    group.append(chosen_segments[i])
                else:
                    grouped.append(group)
                    group = [chosen_segments[i]]
            grouped.append(group)
            return grouped
        return []
    

    def appointment_segment_assign(self,
                                   p: float,
                                   max_chunk_size: int,
                                   segments: Optional[list[list[int]]] = None) -> list[list[int]]:
        """
        Randomly assign appointment time segments from the remaining (unassigned) segments.

        Args:
            p (float): Proportion of remaining segments to sample and assign.
            max_chunk_size (int): The maximum time segment size for each appointment.
            segments (Optional[list[list[int]]], optional): Specific segemnts. Defaults to None.

        Returns:
            list[list[int]]: Newly assigned segments from the remaining pool, grouped consecutively.
        """
        segments = self.segments if segments == None else segments
        segment_n = round(len(segments) * p)

        if segment_n > 0:
            chosen = random.sample(segments, segment_n)
            chosen.sort()

            # First, group into consecutive blocks 
            consecutive_blocks = []
            group = [chosen[0]]
            for i in range(1, len(chosen)):
                if chosen[i] == chosen[i - 1] + 1:
                    group.append(chosen[i])
                else:
                    consecutive_blocks.append(group)
                    group = [chosen[i]]
            consecutive_blocks.append(group)

            # Second, split each consecutive block into random-sized chunks
            appointments = []
            for block in consecutive_blocks:
                i = 0
                while i < len(block):
                    chunk_limit = min(max_chunk_size, len(block) - i)
                    chunk_size = random.randint(1, chunk_limit)
                    appointments.append(block[i:i + chunk_size])
                    i += chunk_size

            return appointments
        return []
        

    def __call__(self,
                 p: float,
                 is_appointment: bool = False,
                 segments: Optional[list[list[int]]] = None,
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
        time_segments = self.appointment_segment_assign(p, segments=segments, **kwargs) \
            if is_appointment else self.schedule_segment_assign(p, segments)

        if len(time_segments):
            return time_segments, [list(convert_segment_to_time(self.start, self.end, self.interval, segments)) for segments in  time_segments]
        return [], []
