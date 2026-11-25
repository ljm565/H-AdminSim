from __future__ import annotations
import os
import math
from dataclasses import replace
from typing import Optional, Sequence, Iterable

from h_adminsim.utils import colorstr
from h_adminsim.policy.base import Policy, Decision
from h_adminsim.registry.errors import PolicyError, PolicyCompositionError, PolicyRequirementError
from h_adminsim.registry.policy import (
    POLICY_HOOKS,
    POLICY_ACTIONS,
    POLICY_BUILTIN_METRIC_IDS,
    POLICY_METRIC_VALUE_TYPES,
    POLICY_METRIC_RESERVED_IDS,
    POLICY_SUPPORTED_OBJECTIVES,
    POLICY_RESERVED_RECORD_KEYS,
    POLICY_REQUIRES_METRIC_PREFIX,   
    POLICY_REQUIRES_PATIENT_PREFIX,
)
from h_adminsim.policy.context import RestrictContext, ReviewContext, _declared
from h_adminsim.policy.metric import Metric
from h_adminsim.policy.evaluation import Evaluation



class PolicyStack:
    """
    The policies of one run, in priority order, and the only object the simulation talks to.

    Several policies may serve the same hook:
        - ``review``: asked in priority order; the first policy that returns a `Decision` decides, and a policy returns
          None to abstain (pass the patient on to the next one). If every policy abstains, the patient keeps their
          preference.
        - ``restrict``: every policy applies; the blocked pairs are combined (union), so order does not matter.

    User-defined metrics (`Metric`) are computed for each negotiation before the `review` policies are asked, and
    every `review` policy sees them in ``ctx.metrics``.

    Outputs are validated here, so a misbehaving policy fails loudly at the call that produced the bad value rather
    than deep inside scheduling.
    """
    def __init__(self,
                 policies: Sequence[Policy] = (),
                 metrics: Sequence[Metric] = (),
                 evaluations: Sequence[Evaluation] = ()):
        """
        Args:
            policies (Sequence[Policy], optional): The run's policies in priority order. A policy without an `id`
                                                   (built without the builder) gets its `name`. Defaults to ().
            metrics (Sequence[Metric], optional): User-defined negotiation metrics, computed in this order. A metric's
                                                  `id` is its key in ``ctx.metrics``. Defaults to ().
            evaluations (Sequence[Evaluation], optional): User-defined evaluations of the finished run (not used during
                                                          the simulation; see `run_evaluations`). Defaults to ().

        Raises:
            PolicyCompositionError: If two policies, metrics or evaluations share an id, or a metric id is a reserved name
                                    (a built-in metric or a record field the simulation owns).
            PolicyRequirementError: If a policy or metric requires a metric that is neither built-in nor listed (for a
                                    metric: listed before it).
        """
        self.policies = list(policies)
        self.metrics = list(metrics)
        self.evaluations = list(evaluations)
        for obj in self.policies + self.metrics + self.evaluations:
            if not obj.id:
                obj.id = obj.name     # built without the builder

        for kind, objs in (('policy', self.policies), ('metric', self.metrics), ('evaluation', self.evaluations)):
            ids = [o.id for o in objs]
            duplicated = sorted({i for i in ids if ids.count(i) > 1})
            if duplicated:
                raise PolicyCompositionError(colorstr("red", f"Duplicate {kind} ids {duplicated}; give each {kind} a unique `id`."))
        
        # Check whether metric ID duplicates with reserved keys
        reserved = sorted(m.id for m in self.metrics if m.id in POLICY_RESERVED_RECORD_KEYS | POLICY_METRIC_RESERVED_IDS)
        if reserved:
            raise PolicyCompositionError(colorstr("red", f"Metric ids {reserved} are reserved names (built-in metrics or record "
                                                         f"fields); choose other ids."))

        # Check that every required metric is computed: built-in, or a user metric (for a metric: one listed before it)
        builtin = POLICY_BUILTIN_METRIC_IDS | {'preference'}
        user = [m.id for m in self.metrics]
        readers = [(p, set(user)) for p in self.policies] + [(m, set(user[:i])) for i, m in enumerate(self.metrics)]
        missing = []
        for obj, visible in readers:
            for m in sorted(_declared(obj, POLICY_REQUIRES_METRIC_PREFIX) - builtin - visible):
                why = 'must be listed before it under `metrics:`' if m in user else 'is neither built-in nor listed under `metrics:`'
                missing.append(f"  - `{obj.id}` requires `metric.{m}`, which {why}")
        if missing:
            lines = '\n'.join(missing)
            raise PolicyRequirementError(colorstr("red", f"Required metrics are not computed:\n{lines}\n"
                                                         f"Built-in metrics: {sorted(POLICY_BUILTIN_METRIC_IDS)}."))

        self._by_hook = {hook: [p for p in self.policies if p.implements(hook)] for hook in POLICY_HOOKS}


    @property
    def reviewers(self) -> list:
        """Policies serving `review`, in priority order."""
        return list(self._by_hook['review'])


    @property
    def restrictors(self) -> list:
        """Policies serving `restrict`, in priority order."""
        return list(self._by_hook['restrict'])
    

    @property
    def required_patient_fields(self) -> dict:
        """
        Patient fields the policies and metrics declared in `requires`.

        Returns:
            dict: ``{field: [ids of the policies / metrics that require it]}``.
        """
        required = {}
        for obj in self.policies + self.metrics:
            for f in sorted(_declared(obj, POLICY_REQUIRES_PATIENT_PREFIX)):
                required.setdefault(f, []).append(obj.id)
        return required


    def find(self, name: str) -> Optional[Policy]:
        """
        The highest-priority policy whose class `name` matches.

        Args:
            name (str): A policy class name (e.g. ``'result_gain'``, ``'capacity_protection'``).

        Returns:
            Optional[Policy]: The first matching policy in priority order, or None if none matches.
        """
        return next((p for p in self.policies if p.name == name), None)


    def restrict_by_policy(self, ctx: RestrictContext) -> dict:
        """
        `(test_code, date)` pairs each `restrict` policy blocks for the patient.

        Args:
            ctx (RestrictContext): Snapshot of the hospital state for the patient about to be scheduled.

        Raises:
            PolicyError: If a policy returns something other than `(test_code, date)` pairs.

        Returns:
            dict: ``{policy id: frozenset of pairs}``, one entry per `restrict` policy in priority order.
        """
        blocked = {}
        for policy in self._by_hook['restrict']:
            pairs = policy.restrict(ctx.for_policy(policy))
            try:
                blocked[policy.id] = frozenset((str(code), str(date)) for code, date in (pairs or ()))
            except (TypeError, ValueError) as e:
                raise PolicyError(colorstr("red", f"Policy `{policy.id}`.restrict must return (test_code, date) pairs: {e}")) from e
        return blocked


    def restrict(self, ctx: RestrictContext) -> frozenset:
        """
        `(test_code, date)` pairs the patient may not book: the union over every `restrict` policy.

        Args:
            ctx (RestrictContext): Snapshot of the hospital state for the patient about to be scheduled.

        Returns:
            frozenset: The blocked `(test_code, date)` pairs (empty without a `restrict` policy).
        """
        return frozenset().union(*self.restrict_by_policy(ctx).values())


    def review(self, ctx: ReviewContext) -> Decision:
        """
        The decision of the first `review` policy (in priority order) that does not abstain, validated and stamped with
        its id; ``'keep'`` with no `decided_by` when there is none or all abstain.

        Args:
            ctx (ReviewContext): Snapshot of the patient's proposed schedule and its negotiation metrics.

        Returns:
            Decision: The decision, with `decided_by` set to the deciding policy's id.
        """
        for policy in self._by_hook['review']:
            decision = policy.review(ctx.for_policy(policy))
            if decision is None:
                continue
            self._validate(decision, policy.id, metric_ids={m.id for m in self.metrics})
            return replace(decision, decided_by=policy.id)
        return Decision.keep()


    def compute_metrics(self, ctx: ReviewContext) -> dict:
        """
        Values of the user-defined metrics for one negotiation, in order.

        Each metric sees the built-in metrics and the user metrics computed before it.

        Args:
            ctx (ReviewContext): The negotiation, with the built-in metrics in ``ctx.metrics``.

        Returns:
            dict: ``{metric id: value}`` (empty without metrics).
        """
        values = {}
        for metric in self.metrics:
            current = replace(ctx, metrics={**ctx.metrics.to_dict(), **values}) if values else ctx
            value = metric.compute(current.for_policy(metric))
            self._validate_metric_value(value, metric.id)
            values[metric.id] = value
        return values


    def metrics_params(self, preference: str) -> dict:
        """
        `NegotiationMetrics` keyword arguments from the `review` policies. The metrics are computed once for all of
        them, so two policies asking for different values of the same parameter cannot run together.

        Args:
            preference (str): The patient's scheduling preference (``'visit_min'``, ``'stay_min'``, ...).

        Raises:
            PolicyCompositionError: If two `review` policies ask for different values of the same parameter.

        Returns:
            dict: The merged keyword arguments (empty without a `review` policy).
        """
        merged, owner = {}, {}
        for policy in self._by_hook['review']:
            for key, value in dict(policy.metrics_params(preference)).items():
                if key in merged and merged[key] != value:
                    raise PolicyCompositionError(colorstr("red",
                        f"Policies `{owner[key]}` and `{policy.id}` need different `{key}` ({merged[key]} vs {value}) for the "
                        f"shared negotiation metrics; set the same value in both."))
                merged.setdefault(key, value)
                owner.setdefault(key, policy.id)
        return merged


    def check_patient_fields(self, available: Iterable[str], where: str = 'agent data'):
        """
        Check that the data provides every patient field the policies require, before any patient is simulated.

        Args:
            available (Iterable[str]): Patient fields the data provides for every follow-up patient
                                       (see `available_patient_fields`).
            where (str, optional): Label of the data (e.g. its path), for the error message. Defaults to 'agent data'.

        Raises:
            PolicyRequirementError: If a required field is missing, listing each one with the policies that need it.
        """
        available = set(available)
        missing = {f: ids for f, ids in self.required_patient_fields.items() if f not in available}
        if missing:
            lines = '\n'.join(f"  - `patient.{f}` (required by {', '.join(f'`{i}`' for i in ids)})" for f, ids in sorted(missing.items()))
            raise PolicyRequirementError(colorstr("red",
                f"{where} does not provide the patient information the policies require:\n{lines}\n"
                f"Add the field(s) to the agent data (or regenerate it) before using these policies. "
                f"Available patient fields: {sorted(available)}."))


    def describe_metrics(self) -> list:
        """
        Run-metadata description of every user-defined metric, recorded under ``_metadata.metrics`` in the result files.

        Returns:
            list: One ``{'id', 'name', 'source', 'source_sha256', 'api_version', 'requires', 'params'}`` per metric.
        """
        return [
            {
                'id': m.id, 'name': m.name, 'source': m.source, 
                'source_sha256': m.source_sha256, 'api_version': m.api_version, 
                'requires': list(m.requires), 'params': m.params()
            } for m in self.metrics
        ]


    def describe_evaluations(self) -> list:
        """
        Run-metadata description of every user-defined evaluation, recorded under ``_metadata.evaluations``.

        Returns:
            list: One ``{'id', 'name', 'source', 'source_sha256', 'api_version', 'params'}`` per evaluation.
        """
        return [
            {
                'id': e.id, 'name': e.name, 'source': e.source, 
                'source_sha256': e.source_sha256, 'api_version': e.api_version, 
                'params': e.params()
            } for e in self.evaluations
        ]


    def describe(self) -> list:
        """
        Run-metadata description of every policy, recorded under ``_metadata.policies`` in the result files.

        Returns:
            list: One ``{'id', 'name', 'source', 'source_sha256', 'api_version', 'hooks', 'requires', 'params'}`` per policy, in
                  priority order.
        """
        return [
            {
                'id': p.id, 'name': p.name, 'source': p.source, 
                'source_sha256': p.source_sha256, 'api_version': p.api_version, 
                'hooks': list(p.hooks()), 'requires': list(p.requires), 'params': p.params()
            } for p in self.policies
        ]


    @staticmethod
    def _validate(decision: Decision, policy_id: str, metric_ids: frozenset = frozenset()):
        """
        Check that a `review` decision is something the simulation can carry out.

        Args:
            decision (Decision): The decision a `review` policy returned (not None; None means abstain).
            policy_id (str): Id of the policy that returned it, for error messages.
            metric_ids (frozenset, optional): Ids of the user-defined metrics, which the record may not overwrite
                                              either. Defaults to an empty set.

        Raises:
            PolicyError: If `decision` is not a `Decision`, has an unknown action, asks for an objective the engine
                         cannot book, chooses ``'negotiate'`` without an existing prompt file, or its `record`
                         overwrites a field the simulation owns or a user-defined metric.
        """
        where = f"Policy `{policy_id}`.review"
        if not isinstance(decision, Decision):
            raise PolicyError(colorstr("red", f"{where} must return a Decision (or None to abstain), got {type(decision).__name__}."))
        if decision.action not in POLICY_ACTIONS:
            raise PolicyError(colorstr("red", f"{where} returned unknown action `{decision.action}`; choose one of {list(POLICY_ACTIONS)}."))
        if decision.action != 'keep' and decision.objective not in POLICY_SUPPORTED_OBJECTIVES:
            raise PolicyError(colorstr("red", f"{where} asked for objective `{decision.objective}`; the simulation can only "
                                              f"book {list(POLICY_SUPPORTED_OBJECTIVES)}."))
        if decision.action == 'negotiate' and not (decision.prompt_path and os.path.isfile(decision.prompt_path)):
            raise PolicyError(colorstr("red", f"{where} chose 'negotiate' without an existing prompt file (got `{decision.prompt_path}`)."))
        clashes = sorted((POLICY_RESERVED_RECORD_KEYS | set(metric_ids)) & set(decision.record))
        if clashes:
            raise PolicyError(colorstr("red", f"{where} record may not overwrite simulation fields or metrics {clashes}."))


    @staticmethod
    def _validate_metric_value(value, metric_id: str):
        """
        Check that a user-defined metric produced a value the result files can store.

        Args:
            value: The computed value.
            metric_id (str): Id of the metric, for the error message.

        Raises:
            PolicyError: If the value is not a bool, int, float (NaN excluded), str or None.
        """
        if not isinstance(value, POLICY_METRIC_VALUE_TYPES) or (isinstance(value, float) and math.isnan(value)):
            raise PolicyError(colorstr("red", f"Metric `{metric_id}` must return a bool, int, float (not NaN), str or None, "
                                              f"got {type(value).__name__} {value!r}."))
