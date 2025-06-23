import random

from utils.random_utils import convert_segment_to_time, convert_time_to_segment



class ScheduleAssigner:
    def __init__(self, start: float, end: float, interval: float):
        self.start = start
        self.end = end
        self.interval = interval
        self.segments = convert_time_to_segment(self.start, self.end, self.interval)


    def schedule_segment_assign(self, p: float) -> list[list[int]]:
        """
        Randomly assign a proportion of available time segments into grouped consecutive blocks.

        This method selects a random subset of segments, where the number of segments is 
        determined by the proportion `p` (e.g., 0.5 means 50% of all segments).
        The selected segments are then grouped into lists of consecutive segment indices.

        Args:
            p (float): Proportion of total segments to assign, between 0 and 1.

        Returns:
            list[list[int]]: A list of groups, where each group is a list of consecutive segment indices.
                            For example, [[0, 1], [3, 4, 5], [7]].

        Example:
            If self.segments = [0, 1, 2, ..., 11] and p = 0.5,
            this function might return something like:
                [[0, 1], [4], [6, 7, 8]]

        Notes:
            - The segment indices are selected randomly each time the function is called.
            - Groups are always composed of consecutive indices from the selected subset.
        """
        segment_n = round(len(self.segments) * p)

        if segment_n > 0:
            # Select random segments
            chosen_segments = random.sample(self.segments, segment_n)
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
        

    def __call__(self, p: float) -> list[list[float]]:
        """
        Generate grouped time ranges by randomly selecting and grouping a proportion of time segments.

        This method allows the ScheduleAssigner instance to be called directly with a proportion `p`.
        It selects a subset of time segments based on `p`, groups them into consecutive segment blocks,
        and converts each group of segment indices into their corresponding time values.

        Args:
            p (float): Proportion of total segments to select, between 0 and 1.

        Returns:
            list[list[float]]: A list of grouped time ranges. Each group is a list of time values (in hours),
                            corresponding to consecutive time segments.
                            For example: [[0.0, 0.5], [2.0, 3.0]].

        Example:
            >>> assigner = ScheduleAssigner(0, 12, 0.5)
            >>> assigner(0.25)
            [[1.0, 1.5], [4.5], [6.0, 6.5]]
        """
        schedule_segments = self.schedule_segment_assign(p)
        if len(schedule_segments):
            return [list(convert_segment_to_time(self.start, self.end, self.interval, segments)) for segments in  schedule_segments]
        return []
