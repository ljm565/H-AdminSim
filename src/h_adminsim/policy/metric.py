from typing import ClassVar, TYPE_CHECKING
from dataclasses import fields, is_dataclass

if TYPE_CHECKING:
    from h_adminsim.policy.context import ReviewContext



class Metric:
    """
    Base class of a user-defined negotiation metric.

    A metric turns the facts of one negotiation (the proposed and the ``throughput_max`` schedules, the built-in
    metrics and the patient record, i.e. a `ReviewContext`) into one value. Metrics are computed after the built-in
    metrics and before the `review` policies are asked, so every `review` policy sees them in ``ctx.metrics[<id>]``,
    and their values are stored in the booking's ``negotiation_metrics`` record for evaluation.

    Like a policy, a metric is a dataclass whose fields are its configuration, declares the patient fields and metrics it
    reads in `requires`, and must be a pure function of the context (no state, no randomness).
    """
    api_version: ClassVar[int] = 1
    name: ClassVar[str] = ''
    requires: ClassVar[tuple] = ()      # what `compute` reads, as `patient.<field>` / `metric.<id>`

    # Set by the builder (see `Policy`)
    id = ''
    source = None
    source_sha256 = None


    def compute(self, ctx: "ReviewContext"):
        """
        The metric's value for one negotiation.

        Args:
            ctx (ReviewContext): The negotiation (``ctx.metrics`` holds the built-in metrics and the user metrics listed
                                 before this one; only the declared ones are readable).

        Returns:
            bool | int | float | str | None: The value.
        """
        raise NotImplementedError


    def params(self) -> dict:
        """
        The metric's configuration (its dataclass fields), as recorded in the run metadata.

        Returns:
            dict: ``{field: value}``.
        """
        if not is_dataclass(self):
            return {}
        return {f.name: getattr(self, f.name) for f in fields(self)}
