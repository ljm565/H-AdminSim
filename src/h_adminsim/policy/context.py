from __future__ import annotations
from types import MappingProxyType
from typing import Iterable, Mapping, Optional
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace

from h_adminsim.utils.common_utils import get_iso_time, str_to_datetime
from h_adminsim.utils.policy_utils import deep_freeze, thaw
from h_adminsim.registry.errors import PolicyRequirementError
from h_adminsim.registry.policy import POLICY_REQUIRES_PATIENT_PREFIX, POLICY_REQUIRES_METRIC_PREFIX



def _freeze(mapping: Optional[Mapping]) -> Mapping:
    """A read-only shallow copy of `mapping`."""
    return MappingProxyType(dict(mapping or {}))



def _declared(obj, prefix: str) -> set:
    """Keys a policy / metric declared in `requires` under `prefix` (``patient.<field>`` -> ``<field>``)."""
    return {r[len(prefix):] for r in getattr(obj, 'requires', ()) or () if r.startswith(prefix)}



class _DeclaredView(MappingABC):
    """
    Read-only view whose reader may only read the keys it declared in `requires` (as ``<prefix><key>``).

    Reading (or testing membership of) any other key raises `PolicyRequirementError`, and iteration yields the declared
    keys only. A declared key missing from the data also raises. The unrestricted view (the one contexts hold before
    `for_policy`) reads every key.
    """
    __slots__ = ('_data', '_allowed', '_owner')
    PREFIX = ''

    def __init__(self, data: Optional[Mapping] = None, allowed: Optional[Iterable[str]] = None, owner: Optional[str] = None):
        """
        Args:
            data (Optional[Mapping], optional): The data (a view of the same class shares its data). Defaults to None.
            allowed (Optional[Iterable[str]], optional): Keys the reader may access; None allows every key.
                                                         Defaults to None.
            owner (Optional[str], optional): Id of the policy / metric reading the view, for error messages.
                                             Defaults to None.
        """
        self._data = data._data if isinstance(data, type(self)) else self._freeze_data(data)
        self._allowed = frozenset(allowed) if allowed is not None else None
        self._owner = owner


    @staticmethod
    def _freeze_data(data: Optional[Mapping]) -> Mapping:
        """The read-only form of the data."""
        return deep_freeze(dict(data or {}))


    def _missing_message(self, who: str, key: str) -> str:
        """Error message for a declared key the data does not have."""
        return f"{who} needs `{self.PREFIX}{key}`, which is not available."


    def _check(self, key: str):
        """
        Check that the reader may read `key` and that the data has it.

        Args:
            key (str): The key being read.

        Raises:
            PolicyRequirementError: If the reader did not declare `key` in `requires`, or the data lacks it.
        """
        who = f"`{self._owner}`" if self._owner else 'A policy'
        if self._allowed is not None and key not in self._allowed:
            raise PolicyRequirementError(
                f"{who} reads `{self.PREFIX}{key}` without declaring it; add '{self.PREFIX}{key}' to its `requires`.")
        if key not in self._data:
            raise PolicyRequirementError(self._missing_message(who, key))


    def __getitem__(self, key: str):
        self._check(key)
        return self._data[key]


    def __contains__(self, key) -> bool:
        if self._allowed is not None and key not in self._allowed:
            self._check(key)
        return key in self._data


    def __iter__(self):
        return (k for k in self._data if self._allowed is None or k in self._allowed)


    def __len__(self) -> int:
        return sum(1 for _ in self)


    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(thaw(dict(self.items())))!r})"


    def restricted_to(self, keys: Iterable[str], owner: Optional[str]):
        """
        A view over the same data that only reads `keys`.

        Args:
            keys (Iterable[str]): Keys the reader declared.
            owner (Optional[str]): Id of the reading policy / metric, for error messages.

        Returns:
            The restricted view (same class as this one).
        """
        return type(self)(self, allowed=keys, owner=owner)


    def to_dict(self) -> dict:
        """The full data in plain (JSON-serializable) form, whatever the view allows."""
        return thaw(self._data)



class PatientView(_DeclaredView):
    """
    Read-only view of one patient's agent-data record, as `ctx.patient`. A policy reads only the fields it declared
    in `requires` (``patient.<field>``); a declared field missing from this patient's record raises, since the data
    must provide it.
    """
    __slots__ = ()
    PREFIX = POLICY_REQUIRES_PATIENT_PREFIX

    def _missing_message(self, who: str, key: str) -> str:
        return (f"{who} needs `patient.{key}`, which this patient's agent data does not have. "
                f"Add the field to the data (or regenerate it) before using this policy.")



class MetricView(_DeclaredView):
    """
    Read-only view of one negotiation's metrics, as `ctx.metrics`: the built-in metrics (``gpc``, ``tcl``, ...), the
    user-defined metrics computed so far and ``preference``. A policy or metric reads only the metrics it declared in
    `requires` (``metric.<id>``) plus ``preference``, which is always readable.
    """
    __slots__ = ()
    PREFIX = POLICY_REQUIRES_METRIC_PREFIX

    @staticmethod
    def _freeze_data(data: Optional[Mapping]) -> Mapping:
        return _freeze(data)    # scalar values: a shallow read-only copy is enough

    def _missing_message(self, who: str, key: str) -> str:
        return (f"{who} needs `metric.{key}`, which this negotiation does not have; list the metric under `metrics:` "
                f"in the policy config (before any metric that reads it).")



@dataclass(frozen=True)
class RestrictContext:
    """
    Read-only snapshot a `restrict` hook sees for the patient about to be scheduled.

    Fields:
        current_time: Hospital time (ISO 8601).
        start_hour: Daily opening hour of the test devices.
        utc_offset: Hospital UTC offset (e.g. ``'+09:00'``).
        test_capacity: ``{(test_code, date): hours}`` total working hours of each test (devices pooled); 0 on closed days.
        test_duration: ``{test_code: hours}`` duration of one test.
        booked_hours: ``{(test_code, date): hours}`` already booked (fixed bookings and follow-up patients of any
            urgency), excluding the patient's own booking when it is being rescheduled.
        patient: The patient's agent-data record (`PatientView`), including ``urgency``; a policy reads only the fields
            it declared in `requires` (see `patient_record` for the fields).
    """
    current_time: str
    start_hour: float
    utc_offset: Optional[str]
    test_capacity: Mapping = field(default_factory=dict)
    test_duration: Mapping = field(default_factory=dict)
    booked_hours: Mapping = field(default_factory=dict)
    patient: Mapping = field(default_factory=dict)


    def __post_init__(self):
        for name in ('test_capacity', 'test_duration', 'booked_hours'):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        if not isinstance(self.patient, PatientView):
            object.__setattr__(self, 'patient', PatientView(self.patient))


    def lead_hours(self, date: str) -> float:
        """
        Hours from now until `date` opens.

        Args:
            date (str): A date (``YYYY-MM-DD``).

        Returns:
            float: Hours until the date's opening hour (negative once it has opened).
        """
        opening = str_to_datetime(get_iso_time(self.start_hour, date, utc_offset=self.utc_offset))
        return (opening - str_to_datetime(self.current_time)).total_seconds() / 3600


    def for_policy(self, policy) -> "RestrictContext":
        """
        The context as `policy` sees it: `patient` restricted to the fields the policy declared in `requires`.

        Args:
            policy (Policy): The policy about to read the context.

        Returns:
            RestrictContext: The restricted copy.
        """
        return replace(self, patient=self.patient.restricted_to(_declared(policy, PatientView.PREFIX), policy.id))


    def to_dict(self) -> dict:
        """
        JSON-serializable form (tuple keys become ``[code, date, hours]`` rows).

        Returns:
            dict: The context's fields.
        """
        return {
            'current_time': self.current_time,
            'start_hour': self.start_hour,
            'utc_offset': self.utc_offset,
            'test_capacity': [[c, d, h] for (c, d), h in self.test_capacity.items()],
            'test_duration': dict(self.test_duration),
            'booked_hours': [[c, d, h] for (c, d), h in self.booked_hours.items()],
            'patient': self.patient.to_dict(),
        }


    @classmethod
    def from_dict(cls, data: dict) -> "RestrictContext":
        """
        Rebuild a context saved by `to_dict`.

        Args:
            data (dict): The saved form. Snapshots taken before `patient` existed keep the urgency at the top level
                         (``urgency``); it is moved into the patient record.

        Returns:
            RestrictContext: The context.
        """
        patient = dict(data.get('patient') or {})
        if 'urgency' in data:
            patient.setdefault('urgency', data['urgency'])
        return cls(
            current_time=data['current_time'],
            start_hour=data['start_hour'],
            utc_offset=data.get('utc_offset'),
            test_capacity={(c, d): h for c, d, h in data.get('test_capacity') or []},
            test_duration=data.get('test_duration') or {},
            booked_hours={(c, d): h for c, d, h in data.get('booked_hours') or []},
            patient=patient,
        )



@dataclass(frozen=True)
class ReviewContext:
    """
    Read-only snapshot a `review` hook sees once the staff proposed a schedule.

    Fields:
        preference: The objective the proposed schedule was built with (``'visit_min'``, ``'stay_min'``, ...).
        metrics: Negotiation metrics of the proposal against the ``throughput_max`` counterfactual (`MetricView`):
            the built-in ones (``gpc``, ``tcl``, ``G``, ``R``, ``U``, ``U_pref``, ``U_thr``; see `NegotiationMetrics`),
            the user-defined ones, and ``preference``. ``gpc`` is ``inf`` for a free win (the patient concedes nothing
            yet gets results sooner). A policy reads only the metrics it declared in `requires` (and ``preference``).
        patient: The patient's agent-data record (`PatientView`), including ``urgency``; a policy reads only the fields
            it declared in `requires` (see `patient_record` for the fields).
        schedules: ``{'preferred': ..., 'hospital': ...}`` — the proposed schedule built with the patient's preference
            and the ``throughput_max`` counterfactual the metrics compare it with (``test_schedule``, ``test_visit_dates``,
            ``idle_waiting_time``, ``all_results_ready_at``, ...), read-only.
    """
    preference: Optional[str]
    metrics: Mapping = field(default_factory=dict)
    patient: Mapping = field(default_factory=dict)
    schedules: Mapping = field(default_factory=dict)


    def __post_init__(self):
        if not isinstance(self.metrics, MetricView):
            public = {k: v for k, v in dict(self.metrics or {}).items() if not k.startswith('_')}
            public.setdefault('preference', self.preference)
            object.__setattr__(self, 'metrics', MetricView(public))
        if not isinstance(self.patient, PatientView):
            object.__setattr__(self, 'patient', PatientView(self.patient))
        object.__setattr__(self, 'schedules', deep_freeze(dict(self.schedules or {})))


    def for_policy(self, policy) -> "ReviewContext":
        """
        The context as `policy` sees it: `patient` and `metrics` restricted to what the policy declared in `requires`
        (``preference`` stays readable).

        Args:
            policy (Policy | Metric): The policy or metric about to read the context.

        Returns:
            ReviewContext: The restricted copy.
        """
        return replace(self,
                       patient=self.patient.restricted_to(_declared(policy, PatientView.PREFIX), policy.id),
                       metrics=self.metrics.restricted_to(_declared(policy, MetricView.PREFIX) | {'preference'}, policy.id))


    def to_dict(self) -> dict:
        """
        JSON-serializable form.

        Returns:
            dict: The context's fields.
        """
        return {'preference': self.preference, 'metrics': self.metrics.to_dict(),
                'patient': self.patient.to_dict(), 'schedules': thaw(self.schedules)}


    @classmethod
    def from_dict(cls, data: dict) -> "ReviewContext":
        """
        Rebuild a context saved by `to_dict`.

        Args:
            data (dict): The saved form (``patient`` / ``schedules`` may be absent; a top-level ``urgency`` from older
                         snapshots is moved into the patient record).

        Returns:
            ReviewContext: The context.
        """
        patient = dict(data.get('patient') or {})
        if 'urgency' in data:
            patient.setdefault('urgency', data['urgency'])
        return cls(preference=data.get('preference'), metrics=data.get('metrics') or {},
                   patient=patient, schedules=data.get('schedules') or {})
