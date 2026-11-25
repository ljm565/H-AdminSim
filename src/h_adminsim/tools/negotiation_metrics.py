import math
from bisect import bisect_right
from collections import defaultdict
from functools import cached_property
from decimal import Decimal, getcontext
from typing import Optional, TYPE_CHECKING

from h_adminsim.utils.common_utils import (
    iso_to_date,
    iso_to_hour,
    get_iso_time,
    str_to_datetime,
    convert_time_to_segment,
)

if TYPE_CHECKING:
    from h_adminsim.tools.scheduling_rule import SchedulingRule
    from h_adminsim.environment.hospital import HospitalEnvironment



def frontload_pairs(booked: set, free: set) -> tuple[int, int]:
    """
    Mann-Whitney pair counts behind the front-loading score (booked slots followed by a free slot).

    `good / total` is the score (1.0 = fully front-loaded, 0.0 = fully back-loaded); counts can be pooled across devices.

    Args:
        booked (set): Booked slot indices on the axis.
        free (set): Free slot indices on the same axis (disjoint from `booked`).

    Returns:
        tuple[int, int]: `(good, total)` — (booked, later-free) pairs and all `len(booked) * len(free)` pairs.
    """
    free_sorted = sorted(free)
    good = sum(len(free_sorted) - bisect_right(free_sorted, b) for b in booked)
    return good, len(booked) * len(free)



class NegotiationMetrics:
    """
    Per-patient negotiation metrics computed at a successful follow-up test scheduling.

    The baseline is the patient's own preference schedule ``P`` (what they were booked into);
    the throughput_max schedule ``T`` is recomputed as the counterfactual the hospital would
    push a negotiation toward. Every metric contrasts ``P`` against ``T`` (see each property):

    - ``gpc``  Gain Per Concession = R / G (result gain per unit concession).
    - ``tcl``  Test Congestion Level over the window [current_time, max(te_pref, te_thr)].
    - ``G``    Preference concession to switch P -> T, in preference-native units.
    - ``R``    Result-time gain (hours) from switching P -> T.
    - ``U``    Per-patient availability proxy (front-loading gain P -> T).

    Slot-level quantities (``tcl``, ``U``) reflect the device state just before this booking commits.
    Whether to negotiate is decided by the policies, not here.
    """

    def __init__(self,
                 preference: str,
                 achieved_schedule: dict,
                 filtered_test_device_information: dict,
                 rule: "SchedulingRule",
                 environment: "HospitalEnvironment",
                 unavailable: Optional[dict] = None,
                 time_budget_s: float = 10.0,
                 tcl_temperature: float = 1.0):
        """
        Store the booking's inputs; metrics are computed lazily.

        Args:
            preference (str): Booking preference ('visit_min', 'stay_min', 'throughput_max' or 'indifferent').
            achieved_schedule (dict): The preference schedule P (`test_schedule`, `test_visit_dates`, `idle_waiting_time`, `all_results_ready_at`).
            filtered_test_device_information (dict): Device schedules, `{'test': {code: {'duration_hour', 'devices': {device: {'schedule': ...}}}}}`.
            rule (SchedulingRule): Rule used to recompute the throughput_max counterfactual T.
            environment (HospitalEnvironment): Supplies `current_time`.
            unavailable (Optional[dict], optional): Patient's test-time unavailability; T is searched under it. Defaults to None.
            time_budget_s (float, optional): Wall-clock cap (seconds) on the throughput_max search. Defaults to 10.0.
            tcl_temperature (float, optional): Softmax temperature for TCL (lower emphasizes the bottleneck test). Defaults to 1.0.
        """
        self.preference = preference
        self._P = achieved_schedule
        self._tdi = filtered_test_device_information
        self._rule = rule
        self._environment = environment
        self._unavailable = unavailable
        self._time_budget_s = time_budget_s
        self._avail_cache = {}
        
        # Hyperparameters
        self.tcl_temperature = tcl_temperature

        # Same time discretization the scheduler used.
        self._start_hour = rule._START_HOUR
        self._end_hour = rule._END_HOUR
        self._time_unit = rule._TIME_UNIT
        self._utc_offset = rule._utc_offset


    # ---- Shared Inputs -------------------------------------------------------
    @property
    def test_codes(self) -> list:
        """
        Distinct test codes in the achieved schedule P, order-preserving.
        """
        return list(dict.fromkeys(t['code'] for t in (self._P.get('test_schedule') or [])))


    def _last_test_end(self, schedule: dict) -> Optional[str]:
        """
        ISO end time of the last test in `schedule` (relies on `test_schedule` being sorted by date/start).

        Args:
            schedule (dict): A schedule with `test_schedule` entries carrying `end` or a `schedule` pair.

        Returns:
            Optional[str]: The ISO end time, or None without tests.
        """
        tests = schedule.get('test_schedule') or []
        if not tests:
            return None
        last = tests[-1]
        end = last.get('end')
        if end is None and last.get('schedule'):
            end = last['schedule'][1]
        return get_iso_time(float(end), last['date'], self._utc_offset) if end is not None else None
    

    @cached_property
    def _T(self) -> dict:
        """
        throughput_max counterfactual schedule T for the same test set (side-effect-free search).

        `test_schedule` is normalized to a start-sorted list with hour-float start/end, matching P.
        """
        schedule = self._rule.schedule_tests(
            'throughput_max', self._tdi, self.test_codes, self._time_budget_s,
            unavailable=self._unavailable,
        )
        schedule['test_schedule'] = sorted(
            ({**t, 'start': iso_to_hour(t['start']), 'end': iso_to_hour(t['end'])}
             for t in schedule['test_schedule'].values()),
            key=lambda x: (x['date'], x['start']),
        )
        return schedule


    @property
    def r_pref(self) -> Optional[str]:
        """
        Result-ready time of the preference schedule P.
        """
        return self._P.get('all_results_ready_at')


    @property
    def r_thr(self) -> Optional[str]:
        """
        Result-ready time of the throughput_max counterfactual T.
        """
        return self._T.get('all_results_ready_at')
    

    @property
    def te_pref(self) -> Optional[str]:
        """
        Last test end time (ISO) of the preference schedule P (a slot window end anchor).
        """
        return self._last_test_end(self._P)


    @property
    def te_thr(self) -> Optional[str]:
        """
        Last test end time (ISO) of the throughput_max counterfactual T.
        """
        return self._last_test_end(self._T)


    # ---- Slot-level Helpers --------------------------------------------------
    def _busy_segments(self, intervals: list) -> set:
        """
        Occupied segment indices of `[start_hour, end_hour]` busy intervals (invalid intervals skipped).

        Args:
            intervals (list): Busy `[start_hour, end_hour]` pairs (None is treated as empty).

        Returns:
            set: The occupied segment indices.
        """
        segs = set()
        for interval in intervals or []:
            s, e = interval[0], interval[1]
            if s is None or e is None or not (float(s) < float(e)):
                continue
            segs.update(convert_time_to_segment(self._start_hour, self._end_hour, self._time_unit, [float(s), float(e)]))
        return segs


    @cached_property
    def _window(self) -> tuple:
        """
        Window bounds (start_date, start_hour, end_date, end_hour) for [current_time, max(te_pref, te_thr)].

        Ends at the last test end (not result-ready); without test ends, at the same day's close.
        """
        ct = self._environment.current_time
        start_date, start_hour = iso_to_date(ct), iso_to_hour(ct)


        test_ends = [t for t in (self.te_pref, self.te_thr) if t]
        if test_ends:
            t1 = max(test_ends, key=str_to_datetime)
            end_date, end_hour = iso_to_date(t1), iso_to_hour(t1)
        else:
            end_date, end_hour = start_date, self._end_hour
        return start_date, start_hour, end_date, end_hour


    def _available_segments(self, date: str) -> set:
        """
        Schedulable segment indices on `date` within the window (first/last days clipped).

        Args:
            date (str): Date (`YYYY-MM-DD`).

        Returns:
            set: The segment indices (empty outside the window).
        """
        if date in self._avail_cache:
            return self._avail_cache[date]
        start_date, start_hour, end_date, end_hour = self._window
        if date < start_date or date > end_date:
            segs = set()
        else:
            lo, hi = self._start_hour, self._end_hour
            if date == start_date:
                lo = max(lo, start_hour)   # first day: drop past hours before current_time
            if date == end_date:
                hi = min(hi, end_hour)     # last day: drop hours beyond t_1
            segs = set(convert_time_to_segment(self._start_hour, self._end_hour, self._time_unit, [lo, hi])) if lo < hi else set()
        self._avail_cache[date] = segs
        return segs


    # ---- Gain Per Concession (GPC) -----------------------------------
    @cached_property
    def R(self) -> float:
        """
        Result-time gain (hours) from switching P -> T, floored at 0 (0.0 if either result time is missing).
        """
        if not self.r_pref or not self.r_thr:
            return 0.0
        delta = (str_to_datetime(self.r_pref) - str_to_datetime(self.r_thr)).total_seconds() / 3600.0
        return max(0.0, delta)


    @cached_property
    def G(self) -> float:
        """
        Preference concession to switch P -> T, floored at 0.

        visit_min: extra visit days; stay_min: extra idle hours; otherwise 0.
        """
        if self.preference == 'visit_min':
            p_days = len(self._P.get('test_visit_dates') or [])
            t_days = len(self._T.get('test_visit_dates') or [])
            return float(max(0, t_days - p_days))
        if self.preference == 'stay_min':
            p_idle = float(self._P.get('idle_waiting_time') or 0.0)
            t_idle = float(self._T.get('idle_waiting_time') or 0.0)
            return max(0.0, t_idle - p_idle)
        return 0.0


    @property
    def gpc(self) -> float:
        """
        Gain Per Concession = R / G: 0.0 if R == 0, inf if G == 0 and R > 0.

        inf is not strict JSON and poisons averages; filter it before aggregating.
        """
        if self.R == 0:
            return 0.0
        if self.G > 0:
            return self.R / self.G 
        return float('inf')


    # ---- Test Congestion Level (TCL) -----------------------------------------
    def _free_slots_for_test(self, code: str) -> int:
        """
        Free device slots for one test across its devices within the window.

        Args:
            code (str): Test code.

        Returns:
            int: Number of free segments.
        """
        test_info = self._tdi.get('test', {}).get(code, {})
        free = 0
        for device_info in test_info.get('devices', {}).values():
            for date, intervals in (device_info.get('schedule') or {}).items():
                avail = self._available_segments(date)
                if avail:
                    free += len(avail - self._busy_segments(intervals))
        return free


    def _demand_slots_for_test(self, code: str) -> int:
        """
        Slots one instance of the test occupies (Q_s, at least 1).

        Args:
            code (str): Test code.

        Returns:
            int: Number of segments.
        """
        getcontext().prec = 10
        dur = self._tdi.get('test', {}).get(code, {}).get('duration_hour', 0) or 0
        return max(1, int(Decimal(str(dur)) / Decimal(str(self._time_unit))))


    @staticmethod
    def _softmax_weighted_mean(values: list, temperature: float) -> float:
        """
        Softmax(values / temperature)-weighted mean of `values`; a lower temperature leans toward the max.

        Args:
            values (list): Non-empty list of values.
            temperature (float): Softmax temperature; <= 0 returns the max.

        Returns:
            float: The weighted mean.
        """
        if temperature <= 0:
            return max(values)
        z = [v / temperature for v in values]
        m = max(z)
        exps = [math.exp(zi - m) for zi in z]
        total = sum(exps)
        return sum((e / total) * v for e, v in zip(exps, values))


    @cached_property
    def tcl(self) -> float:
        """
        Test Congestion Level: softmax-weighted mean of per-test load Q_s / F_s, each capped at 1 (0.0 without tests).
        """
        loads = []
        for code in self.test_codes:
            free = self._free_slots_for_test(code)
            demand = self._demand_slots_for_test(code)
            loads.append(1.0 if free <= 0 else min(1.0, demand / free))
        return self._softmax_weighted_mean(loads, self.tcl_temperature) if loads else 0.0
    

    # ---- Device Utility (U) --------------------------------------------------
    def _patient_slots(self, schedule: dict) -> dict:
        """
        This patient's placements in `schedule`, clipped to the window.

        Args:
            schedule (dict): A schedule whose `test_schedule` entries carry hour-float start/end or a `schedule` pair.

        Returns:
            dict: `{device: {date: set of busy segments}}`.
        """
        out = defaultdict(lambda: defaultdict(set))
        items = schedule.get('test_schedule')
        items = list(items.values()) if isinstance(items, dict) else (items or [])
        for it in items:
            device, date = it.get('device'), it.get('date')
            s, e = it.get('start'), it.get('end')
            if s is None and it.get('schedule'):
                s, e = it['schedule'][0], it['schedule'][1]
            if device is None or date is None or s is None or e is None:
                continue
            segs = self._busy_segments([[float(s), float(e)]]) & self._available_segments(date)
            if segs:
                out[device][date] |= segs
        return out


    def _frontload_score(self, patient_slots: dict) -> float:
        """
        Pooled front-loading score on the patient's devices over the date-flattened window slots.

        Fraction of (booking, later free slot) pairs: 1.0 = fully front, 0.0 = fully back, 0.0 if no pairs.

        Args:
            patient_slots (dict): `{device: {date: set of busy segments}}` from `_patient_slots`.

        Returns:
            float: The score in [0, 1].
        """
        devices = set(patient_slots)

        # Per device: window operating dates (a schedule entry with schedulable slots) + other
        # patients' busy per date. Empty operating dates are kept so they can count toward the axis.
        op_dates = defaultdict(set)
        base = defaultdict(lambda: defaultdict(set))
        for test_info in self._tdi.get('test', {}).values():
            for dname, dinfo in test_info.get('devices', {}).items():
                if dname not in devices:
                    continue
                for date, intervals in (dinfo.get('schedule') or {}).items():
                    if not self._available_segments(date):
                        continue
                    op_dates[dname].add(date)
                    busy = self._busy_segments(intervals) & self._available_segments(date)
                    if busy:
                        base[dname][date] |= busy

        tot_good, tot_max = 0, 0
        for dname in devices:
            dates = sorted(op_dates[dname] | set(patient_slots[dname]))

            # Flatten: assign every schedulable slot a global index, dates concatenated in order.
            offset, rank, acc = {}, {}, 0
            for d in dates:
                av = sorted(self._available_segments(d))
                rank[d] = {seg: i for i, seg in enumerate(av)}
                offset[d] = acc
                acc += len(av)

            gi = lambda d, s: offset[d] + rank[d][s]
            booked = {gi(d, s) for d in dates for s in patient_slots[dname].get(d, set()) if s in rank[d]}
            others = {gi(d, s) for d in dates for s in base[dname].get(d, set()) if s in rank[d]}
            free = set(range(acc)) - others - booked   # slots the patient could have taken but didn't
            if not booked or not free:
                continue

            # Mann-Whitney: for each booking, count free slots that come AFTER it.
            good, total = frontload_pairs(booked, free)
            tot_good += good
            tot_max += total

        return (tot_good / tot_max) if tot_max else 0.0


    @cached_property
    def u_pref(self) -> float:
        """
        Front-loading score of the preference schedule P (1.0 = fully front); 0.0 on failure.
        """
        try:
            return self._frontload_score(self._patient_slots(self._P))
        except Exception:
            return 0.0


    @cached_property
    def u_thr(self) -> float:
        """
        Front-loading score of the throughput_max counterfactual T; 0.0 on failure.
        """
        try:
            return self._frontload_score(self._patient_slots(self._T))
        except Exception:
            return 0.0


    @cached_property
    def U(self) -> float:
        """
        Per-patient availability proxy U = u_thr - u_pref (positive = T is more front-loaded).

        Not additive across patients; the cohort gain needs an ON/OFF A/B simulation.
        """
        return self.u_thr - self.u_pref


    # ---- export --------------------------------------------------------------
    def to_dict(self) -> dict:
        """
        All metrics plus the preference and both schedules as a plain dict.

        Returns:
            dict: `preference`, `gpc`, `tcl`, `G`, `R`, `U`, `U_pref`, `U_thr`, `_P`, `_T`.
        """
        return {
            'preference': self.preference,
            'gpc': self.gpc,
            'tcl': self.tcl,
            'G': self.G,
            'R': self.R,
            'U': self.U,
            'U_pref': self.u_pref,
            'U_thr': self.u_thr,
            '_P': self._P,
            '_T': self._T,
        }
