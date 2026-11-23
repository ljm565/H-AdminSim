from __future__ import annotations
import os
from dataclasses import replace
from typing import Optional, Sequence

from h_adminsim.utils import colorstr
from h_adminsim.policy.base import Policy, Decision
from h_adminsim.registry.errors import PolicyError, PolicyCompositionError
from h_adminsim.registry.policy import (
    POLICY_HOOKS,
    POLICY_ACTIONS,
    POLICY_SUPPORTED_OBJECTIVES,
    POLICY_RESERVED_RECORD_KEYS,
)
from h_adminsim.policy.context import RestrictContext, ReviewContext



class PolicyStack:
    """
    The policies of one run, in priority order, and the only object the simulation talks to.

    Several policies may serve the same hook:
        - ``review``: asked in priority order; the first policy that returns a `Decision` decides, and a policy returns
          None to abstain (pass the patient on to the next one). If every policy abstains, the patient keeps their
          preference.
        - ``restrict``: every policy applies; the blocked pairs are combined (union), so order does not matter.

    Outputs are validated here, so a misbehaving policy fails loudly at the call that produced the bad value rather
    than deep inside scheduling.
    """
    def __init__(self, policies: Sequence[Policy] = ()):
        """
        Args:
            policies (Sequence[Policy], optional): The run's policies in priority order. A policy without an `id`
                                                   (built without the builder) gets its `name`. Defaults to ().

        Raises:
            PolicyCompositionError: If two policies share an id.
        """
        self.policies = list(policies)
        for policy in self.policies:
            if not policy.id:
                policy.id = policy.name     # policies built without the builder

        ids = [p.id for p in self.policies]
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        if duplicated:
            raise PolicyCompositionError(colorstr("red", f"Duplicate policy ids {duplicated}; give each policy a unique `id`."))

        self._by_hook = {hook: [p for p in self.policies if p.implements(hook)] for hook in POLICY_HOOKS}


    @property
    def reviewers(self) -> list:
        """Policies serving `review`, in priority order."""
        return list(self._by_hook['review'])


    @property
    def restrictors(self) -> list:
        """Policies serving `restrict`, in priority order."""
        return list(self._by_hook['restrict'])


    def find(self, name: str) -> Optional[Policy]:
        """
        The highest-priority policy whose class `name` matches.

        Args:
            name (str): A policy class name (e.g. ``'result_gain'``, ``'capacity_protection'``).

        Returns:
            Optional[Policy]: The first matching policy in priority order, or None if none matches.
        """
        return next((p for p in self.policies if p.name == name), None)


    def restrict(self, ctx: RestrictContext) -> frozenset:
        """
        `(test_code, date)` pairs the patient may not book: the union over every `restrict` policy.

        Args:
            ctx (RestrictContext): Snapshot of the hospital state for the patient about to be scheduled.

        Raises:
            PolicyError: If a policy returns something other than `(test_code, date)` pairs.

        Returns:
            frozenset: The blocked `(test_code, date)` pairs (empty without a `restrict` policy).
        """
        blocked = set()
        for policy in self._by_hook['restrict']:
            pairs = policy.restrict(ctx)
            try:
                blocked.update((str(code), str(date)) for code, date in (pairs or ()))
            except (TypeError, ValueError) as e:
                raise PolicyError(colorstr("red", f"Policy `{policy.id}`.restrict must return (test_code, date) pairs: {e}")) from e
        return frozenset(blocked)


    def review(self, ctx: ReviewContext) -> Decision:
        """
        The decision of the first `review` policy (in priority order) that does not abstain, validated and stamped with
        its id; ``'keep'`` with no `decided_by` when there is none or all abstain.

        Args:
            ctx (ReviewContext): Snapshot of the patient's proposed schedule and its negotiation metrics.

        Raises:
            PolicyError: If the deciding policy returns an invalid decision (see `_validate`).

        Returns:
            Decision: The decision, with `decided_by` set to the deciding policy's id.
        """
        for policy in self._by_hook['review']:
            decision = policy.review(ctx)
            if decision is None:
                continue
            self._validate(decision, policy.id)
            return replace(decision, decided_by=policy.id)
        return Decision.keep()


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


    def describe(self) -> list:
        """
        Run-metadata description of every policy, recorded under ``_metadata.policies`` in the result files.

        Returns:
            list: One ``{'id', 'name', 'source', 'source_sha256', 'api_version', 'hooks', 'params'}`` per policy, in
                  priority order.
        """
        return [{'id': p.id, 'name': p.name, 'source': p.source, 'source_sha256': p.source_sha256,
                 'api_version': p.api_version, 'hooks': list(p.hooks()), 'params': p.params()}
                for p in self.policies]


    @staticmethod
    def _validate(decision: Decision, policy_id: str):
        """
        Check that a `review` decision is something the simulation can carry out.

        Args:
            decision (Decision): The decision a `review` policy returned (not None; None means abstain).
            policy_id (str): Id of the policy that returned it, for error messages.

        Raises:
            PolicyError: If `decision` is not a `Decision`, has an unknown action, asks for an objective the engine
                         cannot book, chooses ``'negotiate'`` without an existing prompt file, or its `record`
                         overwrites a field the simulation owns.
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
        clashes = sorted(POLICY_RESERVED_RECORD_KEYS & set(decision.record))
        if clashes:
            raise PolicyError(colorstr("red", f"{where} record may not overwrite simulation fields {clashes}."))
