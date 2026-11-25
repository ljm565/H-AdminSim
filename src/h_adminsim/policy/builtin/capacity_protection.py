from typing import ClassVar
from dataclasses import dataclass

from h_adminsim.registry.errors import PolicyError
from h_adminsim.policy.base import Policy
from h_adminsim.policy.context import RestrictContext
from h_adminsim.utils import colorstr



@dataclass
class CapacityProtectionPolicy(Policy):
    """
    Airline-style capacity protection of follow-up tests for `urgent` patients (two-class booking limit).

    Each test (all of its devices pooled) keeps a `ratio` share of its total capacity on a date (working hours of its
    open devices) for `urgent` patients: a `normal` patient may book the test on that date only while the hours already
    booked there (fixed bookings and patients of any urgency) plus the new test stay within `(1 - ratio)` of that
    capacity. `urgent` patients may use every free slot, so their bookings also consume the protected share (nested
    protection).

    Only dates opening within `horizon_days` of the current time are protected, and a date is released to everyone
    once it opens within `release_hours`, so held-back capacity that urgent demand did not take is not wasted.

    Fields:
        ratio: Share of each (test, date) total capacity held back for `urgent` patients, in [0, 1).
        horizon_days: Protect dates opening within this many days of the current time.
        release_hours: Release a date to `normal` patients once it opens within this many hours.
    """
    name: ClassVar[str] = 'capacity_protection'
    requires: ClassVar[tuple] = ('patient.urgency',)

    ratio: float = 0.2
    horizon_days: float = 3.0
    release_hours: float = 24.0


    def __post_init__(self):
        """
        Check the parameters when the policy is built.

        Raises:
            PolicyError: If `ratio` is not in [0, 1), `horizon_days` is not positive, or `release_hours` is not in
                         [0, horizon_days * 24).
        """
        if not 0 <= self.ratio < 1:
            raise PolicyError(colorstr("red", f"`ratio` must be in [0, 1), got {self.ratio}."))
        if self.horizon_days <= 0:
            raise PolicyError(colorstr("red", f"`horizon_days` must be > 0, got {self.horizon_days}."))
        if not 0 <= self.release_hours < self.horizon_days * 24:
            raise PolicyError(colorstr("red", f"`release_hours` must be in [0, horizon_days * 24), got {self.release_hours}."))


    def is_protected(self, lead_hours: float) -> bool:
        """
        Whether a date whose opening is `lead_hours` away from now is still held back for `urgent` patients.

        Args:
            lead_hours (float): Hours until the date opens.

        Returns:
            bool: True if the date is protected.
        """
        return self.release_hours <= lead_hours < self.horizon_days * 24


    def normal_limit(self, capacity_hours: float) -> float:
        """
        Total booked hours of a (test, date) up to which `normal` patients may still book.

        Args:
            capacity_hours (float): Total working hours of the test on the date.

        Returns:
            float: The booking limit in hours.
        """
        return (1 - self.ratio) * capacity_hours


    def restrict(self, ctx: RestrictContext) -> set:
        """
        A `normal` patient is blocked from a protected date of a test once the hours already booked there plus this
        test's duration would exceed the booking limit; `urgent` patients are never blocked. Closed days (no capacity)
        are skipped, as nobody can book them.

        Args:
            ctx (RestrictContext): Snapshot for the patient about to be scheduled.

        Returns:
            set: Blocked ``(test_code, date)`` pairs.
        """
        blocked = set()
        if ctx.patient['urgency'] != 'normal':
            return blocked
        for (code, date), capacity_hours in ctx.test_capacity.items():
            if capacity_hours <= 0 or not self.is_protected(ctx.lead_hours(date)):
                continue
            if ctx.booked_hours.get((code, date), 0.0) + ctx.test_duration[code] > self.normal_limit(capacity_hours) + 1e-9:
                blocked.add((code, date))
        return blocked
