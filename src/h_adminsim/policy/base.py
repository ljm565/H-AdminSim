from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
from typing import ClassVar, Iterable, Mapping, Optional, TYPE_CHECKING

from h_adminsim.registry.policy import POLICY_HOOKS

if TYPE_CHECKING:
    from h_adminsim.policy.context import RestrictContext, ReviewContext



@dataclass(frozen=True)
class Decision:
    """
    Outcome of a `review` hook for one patient.

    Fields:
        action: ``'keep'`` (book the patient's own preference), ``'auto'`` (book `objective` directly, without
            persuasion) or ``'negotiate'`` (run the persuasion sub-loop toward `objective`).
        objective: Scheduling objective booked on ``'auto'`` / pushed on ``'negotiate'`` (None on ``'keep'``).
        prompt_path: Staff persuasion system prompt for ``'negotiate'``.
        record: Extra fields stored on the booking's `negotiation_metrics` (e.g. ``{'ti': 0.8}``).
        decided_by: Id of the policy that made the decision (filled in by `PolicyStack`).
    """
    action: str
    objective: Optional[str] = None
    prompt_path: Optional[str] = None
    record: Mapping = field(default_factory=dict)
    decided_by: Optional[str] = None


    @classmethod
    def keep(cls, **record) -> "Decision":
        return cls('keep', record=record)


    @classmethod
    def auto(cls, objective: str = 'throughput_max', **record) -> "Decision":
        return cls('auto', objective=objective, record=record)


    @classmethod
    def negotiate(cls, prompt_path: str, objective: str = 'throughput_max', **record) -> "Decision":
        return cls('negotiate', objective=objective, prompt_path=prompt_path, record=record)


    @property
    def do_negotiate(self) -> bool:
        return self.action == 'negotiate'



class Policy:
    """
    Base class of every follow-up scheduling policy.

    A policy plugs into the simulation only through hooks, each optional:
        - ``restrict(ctx) -> iterable of (test_code, date)``: pairs the current patient may not book. Every
          scheduler call made for the patient (staff tools, negotiation counterfactual, sanity checker,
          rescheduling) sees the same restriction.
        - ``review(ctx) -> Decision | None``: what to do with a patient whose preferred schedule differs from the
          hospital-preferred one. Return None to abstain and let the next `review` policy (in priority order) decide.

    A run may list several policies per hook (see `PolicyStack`): `review` policies are asked in priority order until
    one decides, and every `restrict` policy applies.

    Hooks receive read-only snapshots (`RestrictContext`, `ReviewContext`) and must be pure functions of them:
    a policy keeps no state across patients, since patients are split across worker processes and runs resume.
    Configuration lives in the policy's dataclass fields, which `params()` reports for the run metadata.

    Patient information comes from the agent data through `ctx.patient`, and a policy must declare every patient field
    it reads in `requires` (``('patient.department', ...)``). Likewise, a `review` policy declares the negotiation
    metrics it reads from `ctx.metrics` (``('metric.gpc', ...)``; ``preference`` needs no declaration). A run whose
    agent data lacks a declared field, or that computes no declared metric, fails before it starts, and reading an
    undeclared field or metric raises `PolicyRequirementError`.
    """
    api_version: ClassVar[int] = 1
    name: ClassVar[str] = ''
    requires: ClassVar[tuple] = ()      # what the hooks read, as `patient.<field>` / `metric.<id>`

    # Set by the builder; plain class attributes, not dataclass fields, so `params()` reports configuration only.
    id = ''                 # instance id within a run (defaults to `name`)
    source = None           # `builtin:<name>` or `file:<abs path>:<ClassName>`
    source_sha256 = None    # sha256 of the source file (file sources only)


    @classmethod
    def implements(cls, hook: str) -> bool:
        """
        Whether this policy class overrides `hook`.

        Args:
            hook (str): A hook name (``'restrict'`` or ``'review'``).

        Raises:
            ValueError: If `hook` is not a hook name.

        Returns:
            bool: True if the class overrides the hook.
        """
        if hook not in POLICY_HOOKS:
            raise ValueError(f"Unknown policy hook `{hook}`; choose one of {list(POLICY_HOOKS)}")
        return getattr(cls, hook) is not getattr(Policy, hook)


    @classmethod
    def hooks(cls) -> tuple:
        return tuple(h for h in POLICY_HOOKS if cls.implements(h))


    def restrict(self, ctx: "RestrictContext") -> Iterable[tuple]:
        raise NotImplementedError


    def review(self, ctx: "ReviewContext") -> Optional[Decision]:
        raise NotImplementedError


    def metrics_params(self, preference: str) -> dict:
        """
        Keyword arguments this policy passes to `NegotiationMetrics` for a patient of `preference` (e.g. the TCL
        temperature). Read from `review` policies only.
        """
        return {}


    def params(self) -> dict:
        """The policy's configuration (its dataclass fields), as recorded in the run metadata."""
        if not is_dataclass(self):
            return {}
        return {f.name: getattr(self, f.name) for f in fields(self)}
