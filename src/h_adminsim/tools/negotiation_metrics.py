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
from h_adminsim.utils import colorstr

if TYPE_CHECKING:
    from h_adminsim.tools.scheduling_rule import SchedulingRule
    from h_adminsim.environment.hospital import HospitalEnvironment



class NegotiationMetrics:
    """
    Per-patient negotiation metrics computed at a successful follow-up test scheduling.

    The baseline is the patient's own preference schedule ``P`` (what they were booked into);
    the throughput_max schedule ``T`` is recomputed as the counterfactual the hospital would
    push a negotiation toward. Every metric contrasts ``P`` against ``T`` (see each property):

    - ``pci``  Preference Concession Index = R / G (result gain per unit concession).
    - ``tcl``  Test Congestion Level over the window [current_time, max(te_pref, te_thr)].
    - ``G``    Preference concession to switch P -> T, in preference-native units.
    - ``R``    Result-time gain (hours) from switching P -> T.
    - ``friction`` Dialogue turns spent with this patient (friction proxy).
    - ``U``    Per-patient availability proxy (front-loading gain P -> T).

    Slot-level quantities (``tcl``, ``U``) are read off the same ``filtered_test_device_information``
    the scheduler used, so they reflect the device state just before this booking commits.
    """

    def __init__(self,
                 preference: str,
                 achieved_schedule: dict,
                 filtered_test_device_information: dict,
                 rule: "SchedulingRule",
                 environment: "HospitalEnvironment",
                 dialog_history: Optional[list] = None,
                 time_budget_s: float = 10.0,
                 tcl_temperature: float = 1.0,
                 trigger_temperature: float = 1.0,
                 negotiation_trigger_threshold: float = 1.0):
        """
        Args:
            preference (str): The booking's scheduling preference ('visit_min', 'stay_min',
                              'throughput_max', or 'indifferent'; the latter is treated as throughput_max).
            achieved_schedule (dict): The preference schedule P the patient was booked into
                                      (the `pred_schedule` shape: `test_schedule`, `test_visit_dates`,
                                      `idle_waiting_time`, `all_results_ready_at`).
            filtered_test_device_information (dict): Device schedules keyed as
                                                     `{'test': {code: {'duration_hour', 'devices': {device: {'schedule': {date: [[s, e], ...]}}}}}}`.
            rule (SchedulingRule): Rule instance used to recompute the throughput_max counterfactual T.
            environment (HospitalEnvironment): Environment supplying `current_time`.
            dialog_history (Optional[list], optional): Turn list `[{'role', 'content'}, ...]` for the friction proxy. Defaults to None.
            time_budget_s (float, optional): Wall-clock cap on the throughput_max backtracking search. Defaults to 10.0.
            tcl_temperature (float, optional): Softmax temperature for TCL (lower emphasizes the bottleneck test). Defaults to 1.0.
            trigger_temperature (float, optional): Temperature τ dividing the trigger index
                                                   `ti = PCI * TCL / τ`; a smaller τ raises `ti`, triggering
                                                   negotiation more readily (more aggressive). Must be > 0. Defaults to 1.0.
            negotiation_trigger_threshold (float, optional): Cutoff P* on the trigger index; the patient is
                                                             flagged for negotiation when `ti >= P*`. Higher is
                                                             stricter (fewer negotiations); clamped to >= 0. Defaults to 1.0.
        """
        self.preference = 'throughput_max' if preference in {'throughput_max', 'indifferent'} else preference
        self._P = achieved_schedule
        self._tdi = filtered_test_device_information
        self._rule = rule
        self._environment = environment
        self._dialog_history = dialog_history or []
        self._time_budget_s = time_budget_s
        self._avail_cache = {}
        
        # Hyperparameters
        if trigger_temperature <= 0:
            raise ValueError(colorstr('red', f'trigger_temperature must be > 0 (it divides the trigger index PCI*TCL), got {trigger_temperature}.'))
        self.tcl_temperature = tcl_temperature
        self.trigger_temperature = trigger_temperature
        self.negotiation_trigger_threshold = max(0.0, negotiation_trigger_threshold)

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
        ISO end time of the last test in `schedule` (None if it has no tests). `test_schedule` is
        sorted by (date, start) and the patient's tests never overlap, so the last entry is the
        latest-ending one; its end is its `date` + hour-float `end` (or a `schedule` [start, end] pair).
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
        throughput_max counterfactual schedule for the same test set (side-effect-free search).
        `test_schedule` is normalized to a start-sorted list with hour-float start/end, matching P.
        """
        schedule = self._rule.schedule_tests(
            'throughput_max', self._tdi, self.test_codes, self._time_budget_s
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
        Last test end time (ISO) of the preference schedule P — the slot window's end anchor
        (device slots free at test end, unlike result-ready which comes later).
        """
        return self._last_test_end(self._P)


    @property
    def te_thr(self) -> Optional[str]:
        """
        Last test end time (ISO) of the throughput_max counterfactual T.
        """
        return self._last_test_end(self._T)


    # ---- Slot-level Helpers --------------------------------------------------
    def _busy_segments(self, intervals) -> set:
        """
        Set of occupied segment indices for a list of `[start_hour, end_hour]` busy intervals.
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
        Window bounds (start_date, start_hour, end_date, end_hour) for
        [current_time, max(te_pref, te_thr)] — ending at the later schedule's LAST TEST END (when the
        device slots free), not result-ready (which comes later). Falls back to a same-day window
        ending at the working-day close when no test end is available.
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
        Schedulable segment indices on `date` within the window: the first day is clipped to
        [current_hour, END], the last day to [START, t_1_hour], middle days span the full working
        day. Empty when `date` is outside the window or the clipped range is empty.
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


    # ---- Preference Concession Index (PCI) -----------------------------------
    @cached_property
    def R(self) -> float:
        """
        Result-time gain (hours) from switching P -> T: hours(R_pref) - hours(R_thr),
        floored at 0 (throughput never finishes later than the preference schedule).
        """
        if not self.r_pref or not self.r_thr:
            return 0.0
        delta = (str_to_datetime(self.r_pref) - str_to_datetime(self.r_thr)).total_seconds() / 3600.0
        return max(0.0, delta)


    @cached_property
    def G(self) -> float:
        """
        Preference concession to switch P -> T, in preference-native units and floored at 0:
        visit_min -> extra visit days; stay_min -> extra idle hours; throughput_max -> 0 (already at T).
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
    def pci(self) -> float:
        """
        Preference Concession Index = R / G (result-hours gained per unit concession).

        Three regimes:
        - R == 0        -> 0.0  : no result-time gain, so negotiating buys nothing (regardless of G).
        - G > 0         -> R / G: the usual benefit-per-concession trade-off.
        - G == 0, R > 0 -> inf  : throughput is strictly earlier at zero concession (free win), so it
                                  must be negotiated. NOTE: inf is not strict-JSON and poisons averages
                                  -> treat these as a separate 'dominant' bucket / filter before aggregating.
        """
        if self.R == 0:
            return 0.0
        if self.G > 0:
            return self.R / self.G 
        return float('inf')


    # ---- Test Congestion Level (TCL) -----------------------------------------
    def _free_slots_for_test(self, code: str) -> int:
        """
        Free device slots for one test across its compatible devices within the window
        (boundary days clipped to schedulable hours).
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
        Slots one instance of the test occupies (Q_s).
        """
        getcontext().prec = 10
        dur = self._tdi.get('test', {}).get(code, {}).get('duration_hour', 0) or 0
        return max(1, int(Decimal(str(dur)) / Decimal(str(self._time_unit))))


    @staticmethod
    def _softmax_weighted_mean(values: list, temperature: float) -> float:
        """
        Softmax(values / temperature)-weighted mean of `values` (numerically stabilized).
        Highlights the bottleneck as the temperature rises.
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
        Test Congestion Level: softmax-weighted mean of per-test load L_s = Q_s / F_s over the
        window, clamped to (0, 1] (the window always holds the booked schedule). 0.0 when no tests.
        """
        loads = []
        for code in self.test_codes:
            free = self._free_slots_for_test(code)
            demand = self._demand_slots_for_test(code)
            loads.append(1.0 if free <= 0 else min(1.0, demand / free))
        return self._softmax_weighted_mean(loads, self.tcl_temperature) if loads else 0.0
    

    # ---- Trigger Index (TI) --------------------------------------------------
    @property
    def ti(self) -> float:
        """
        Trigger Index `ti = PCI * TCL / trigger_temperature`: how worth negotiating this patient is,
        combining concession-worthiness (PCI) with device congestion (TCL). The patient is flagged for
        negotiation when `ti >= negotiation_trigger_threshold` (see `do_negotiate` in `to_dict`).
        Inherits PCI's `inf` for a free dominant win (G == 0, R > 0), which always triggers.
        """
        return self.pci * self.tcl / self.trigger_temperature


    # ---- Device Utility (U) --------------------------------------------------
    def _patient_slots(self, schedule: dict) -> dict:
        """
        `{device: {date: set(busy_segments)}}` for this patient's placements in `schedule`.

        Both P and T carry start/end as hour floats here (T is normalized to floats in `_T`); a
        placement may instead store them as a `schedule` [start, end] pair, which is also handled.
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
        Pooled front-loading score (Mann-Whitney compactness) on the patient's devices, over a
        per-device timeline flattened across all window dates. Among the slots available to the patient
        (window slots minus other patients' bookings), it is the fraction of (booking-before-free)
        pairs: 1.0 = every booking precedes every free slot (packed at the earliest days/hours, leaving
        a clean tail), 0.0 = fully back-loaded, 0.5 = interleaved. Higher is better. Since the booking
        count and free count are identical for P and T (same test set), the two share a denominator, so
        score(T) - score(P) is a clean comparison.
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
            free_sorted = sorted(free)
            tot_good += sum(len(free_sorted) - bisect_right(free_sorted, b) for b in booked)
            tot_max += len(booked) * len(free)

        return (tot_good / tot_max) if tot_max else 0.0


    @cached_property
    def U(self) -> float:
        """
        Per-patient availability proxy: front-loading gain from switching P -> T, in isolation.
        Each schedule's front-loading score (Mann-Whitney compactness over the date-flattened slot
        axis; 1.0 = fully front, 0.0 = fully back) is measured, and U = score(T) - score(P). Positive =
        T books earlier and tighter, leaving a larger contiguous tail for later patients.

        NOTE: this is an isolated per-patient proxy. The true cohort availability gain is non-additive
        (patients share the slot pool) and must be measured by an ON/OFF full-simulation A/B; do not
        sum this across patients.
        """
        try:
            score_p = self._frontload_score(self._patient_slots(self._P))
            score_t = self._frontload_score(self._patient_slots(self._T))
            return score_t - score_p
        except Exception:
            return 0.0


    # ---- Dialogue Friction (F) -----------------------------------------------
    @property
    def friction(self) -> int:
        """
        Friction proxy: number of patient utterances in the scheduling dialogue.
        """
        return sum(1 for m in self._dialog_history if m.get('role') == 'Patient')


    # ---- export --------------------------------------------------------------
    def to_dict(self) -> dict:
        """
        All six metrics (plus preference) as a plain dict.
        """
        return {
            'preference': self.preference,
            'pci': self.pci,
            'tcl': self.tcl,
            'ti': self.ti,
            'do_negotiate': self.ti >= self.negotiation_trigger_threshold,
            'G': self.G,
            'R': self.R,
            'F': self.friction,
            'U': self.U,
        }
