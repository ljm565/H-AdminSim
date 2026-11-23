from __future__ import annotations
from types import MappingProxyType
from dataclasses import dataclass, field
from typing import Mapping, Optional

from h_adminsim.utils.common_utils import get_iso_time, str_to_datetime



def _freeze(mapping: Optional[Mapping]) -> Mapping:
    return MappingProxyType(dict(mapping or {}))



@dataclass(frozen=True)
class RestrictContext:
    """
    Read-only snapshot a `restrict` hook sees for the patient about to be scheduled.

    Fields:
        current_time: Hospital time (ISO 8601).
        urgency: The patient's urgency (``'urgent'`` / ``'normal'``, or None if unknown).
        start_hour: Daily opening hour of the test devices.
        utc_offset: Hospital UTC offset (e.g. ``'+09:00'``).
        test_capacity: ``{(test_code, date): hours}`` total working hours of each test (devices pooled); 0 on closed days.
        test_duration: ``{test_code: hours}`` duration of one test.
        booked_hours: ``{(test_code, date): hours}`` already booked (fixed bookings and follow-up patients of any
            urgency), excluding the patient's own booking when it is being rescheduled.
    """
    current_time: str
    urgency: Optional[str]
    start_hour: float
    utc_offset: Optional[str]
    test_capacity: Mapping = field(default_factory=dict)
    test_duration: Mapping = field(default_factory=dict)
    booked_hours: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for name in ('test_capacity', 'test_duration', 'booked_hours'):
            object.__setattr__(self, name, _freeze(getattr(self, name)))


    def lead_hours(self, date: str) -> float:
        """Hours from now until `date` opens (negative once it has opened)."""
        opening = str_to_datetime(get_iso_time(self.start_hour, date, utc_offset=self.utc_offset))
        return (opening - str_to_datetime(self.current_time)).total_seconds() / 3600


    def to_dict(self) -> dict:
        """JSON-serializable form (tuple keys become ``[code, date, hours]`` rows)."""
        return {
            'current_time': self.current_time,
            'urgency': self.urgency,
            'start_hour': self.start_hour,
            'utc_offset': self.utc_offset,
            'test_capacity': [[c, d, h] for (c, d), h in self.test_capacity.items()],
            'test_duration': dict(self.test_duration),
            'booked_hours': [[c, d, h] for (c, d), h in self.booked_hours.items()],
        }


    @classmethod
    def from_dict(cls, data: dict) -> "RestrictContext":
        return cls(
            current_time=data['current_time'],
            urgency=data.get('urgency'),
            start_hour=data['start_hour'],
            utc_offset=data.get('utc_offset'),
            test_capacity={(c, d): h for c, d, h in data.get('test_capacity') or []},
            test_duration=data.get('test_duration') or {},
            booked_hours={(c, d): h for c, d, h in data.get('booked_hours') or []},
        )



@dataclass(frozen=True)
class ReviewContext:
    """
    Read-only snapshot a `review` hook sees once the staff proposed a schedule.

    Fields:
        preference: The objective the proposed schedule was built with (``'visit_min'``, ``'stay_min'``, ...).
        urgency: The patient's urgency (``'urgent'`` / ``'normal'``, or None if unknown).
        metrics: Negotiation metrics of the proposal against the ``throughput_max`` counterfactual
            (``gpc``, ``tcl``, ``G``, ``R``, ``U``, ``U_pref``, ``U_thr``; see `NegotiationMetrics`).
            ``gpc`` is ``inf`` for a free win (the patient concedes nothing yet gets results sooner).
    """
    preference: Optional[str]
    urgency: Optional[str] = None
    metrics: Mapping = field(default_factory=dict)

    def __post_init__(self):
        public = {k: v for k, v in dict(self.metrics or {}).items() if not k.startswith('_')}
        public.setdefault('preference', self.preference)
        object.__setattr__(self, 'metrics', _freeze(public))


    def to_dict(self) -> dict:
        return {'preference': self.preference, 'urgency': self.urgency, 'metrics': dict(self.metrics)}


    @classmethod
    def from_dict(cls, data: dict) -> "ReviewContext":
        return cls(preference=data.get('preference'), urgency=data.get('urgency'), metrics=data.get('metrics') or {})
