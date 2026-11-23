from __future__ import annotations
from typing import Optional

from h_adminsim.utils import colorstr
from h_adminsim.registry.errors import PolicyError



def run_policies(metadata: Optional[dict], where: str = 'run') -> list:
    """
    The policy list a simulation run recorded under ``_metadata.policies`` (see `PolicyStack.describe`).

    Args:
        metadata (Optional[dict]): The run's ``_metadata`` block.
        where (str, optional): Label of the run (e.g. its result file), prefixed to error messages. Defaults to 'run'.

    Raises:
        PolicyError: If the run uses the pre-plugin result format (no ``policies`` key, or the old ``policy`` block is
                     still present), or recorded no policies (it ran no follow-up scheduling task).

    Returns:
        list: One ``{'id', 'name', 'source', 'source_sha256', 'api_version', 'hooks', 'params'}`` per policy, in
              priority order.
    """
    metadata = metadata or {}
    if 'policies' not in metadata:
        raise PolicyError(colorstr("red",
            f"{where}: `_metadata` has no `policies` (pre-plugin result format with only `policy`). "
            f"Convert the result to the current format before evaluating it."))
    if 'policy' in metadata:
        raise PolicyError(colorstr("red",
            f"{where}: `_metadata` still has the pre-plugin `policy` block. The current format records only "
            f"`policies`; remove `policy` when converting the result."))
    if metadata['policies'] is None:
        raise PolicyError(colorstr("red", f"{where}: `_metadata.policies` is empty; the run has no follow-up scheduling task."))
    return list(metadata['policies'])



def run_policy_summary(metadata: Optional[dict], where: str = 'run') -> dict:
    """
    The negotiation mode and urgent capacity protection of a run, read from its ``_metadata.policies``.

    The review policy is the highest-priority `result_gain` one, else the highest-priority policy serving `review`;
    the protection is the highest-priority `capacity_protection` one. A run without a `review` policy keeps every
    patient on their preference, so its mode is ``'patient-side'``.

    Args:
        metadata (Optional[dict]): The run's ``_metadata`` block.
        where (str, optional): Label of the run (e.g. its result file), prefixed to error messages. Defaults to 'run'.

    Raises:
        PolicyError: If the run uses the pre-plugin result format or recorded no policies (see `run_policies`).

    Returns:
        dict: ``{'mode', 'review', 'capacity_protection', 'policies'}`` — ``mode`` (str or None for a custom review
              policy without one), ``review`` (the review policy's entry or None), ``capacity_protection`` (its params
              or None) and the full ``policies`` list.
    """
    policies = run_policies(metadata, where)
    reviewers = [p for p in policies if 'review' in (p.get('hooks') or [])]
    review = next((p for p in reviewers if p.get('name') == 'result_gain'), reviewers[0] if reviewers else None)
    protection = next((p for p in policies if p.get('name') == 'capacity_protection'), None)
    return {
        'mode': (review.get('params') or {}).get('mode') if review else 'patient-side',
        'review': review,
        'capacity_protection': dict(protection.get('params') or {}) if protection else None,
        'policies': policies,
    }
