from types import MappingProxyType
from collections.abc import Mapping as MappingABC
from typing import TYPE_CHECKING, Any, Iterable, Optional

from h_adminsim.registry.errors import PolicyError
from h_adminsim.registry.policy import POLICY_PATIENT_SYMPTOM_FIELDS
from h_adminsim.utils import colorstr
from h_adminsim.utils.filesys_utils import get_files, json_load

if TYPE_CHECKING:
    from h_adminsim.policy.stack import PolicyStack




def deep_freeze(value: Any) -> Any:
    """
    A read-only deep copy of a JSON-like value: dicts become read-only mappings and lists become tuples.

    Args:
        value (Any): Any JSON-like value.

    Returns:
        Any: The frozen value (scalars are returned as they are).
    """
    if isinstance(value, MappingABC):
        return MappingProxyType({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(v) for v in value)
    return value



def thaw(value: Any) -> Any:
    """
    The plain (JSON-serializable) form of a value frozen by `deep_freeze`.

    Args:
        value (Any): A frozen value.

    Returns:
        Any: The value with read-only mappings turned into dicts and tuples into lists.
    """
    if isinstance(value, MappingABC):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value



def patient_record(gt: dict, test_data: Optional[dict] = None) -> dict:
    """
    The patient record a policy sees as `ctx.patient`: the follow-up patient's ground-truth entry plus the
    ``constraint.symptom`` fields listed in `POLICY_PATIENT_SYMPTOM_FIELDS` (``urgency``, ``disease``, ``symptoms``) when present.

    Args:
        gt (dict): The patient's ground-truth entry in the agent data.
        test_data (Optional[dict], optional): The patient's test data (holds ``constraint.symptom``). Defaults to None.

    Returns:
        dict: The patient record.
    """
    record = dict(gt)
    symptom = ((test_data or {}).get('constraint') or {}).get('symptom') or {}
    for key, name in POLICY_PATIENT_SYMPTOM_FIELDS.items():
        if key in symptom:
            record[name] = symptom[key]
    return record



def followup_patient_records(agent_data: dict) -> dict:
    """
    Patient records of every follow-up patient in one hospital's agent data.
    Covers the ``agent_data`` list and the ``_pending_followups`` / ``_ready_followups`` pools built while running.

    Args:
        agent_data (dict): One hospital's agent data.

    Returns:
        dict: ``{patient name: patient record}`` (see `patient_record`), each patient once.
    """
    pools = list(agent_data.get('agent_data') or []) + list((agent_data.get('_pending_followups') or {}).values()) \
        + list(agent_data.get('_ready_followups') or [])
    
    records = {}
    for gt, test_data in pools:
        if gt.get('visit_type') == 'follow_up_visit' and gt.get('patient') not in records:
            records[gt.get('patient')] = patient_record(gt, test_data)
    
    return records



def available_patient_fields(records: Iterable[dict]) -> set:
    """
    Patient fields present in every record, i.e. the fields a policy may require.

    Args:
        records (Iterable[dict]): Patient records (see `patient_record`).

    Returns:
        set: The common fields (empty when there is no record).
    """
    fields = None
    for record in records:
        fields = set(record) if fields is None else fields & set(record)
    return fields or set()



def check_policy_requirements(policies: 'PolicyStack', agent_data_path: str) -> None:
    """
    Check before a run that every agent data file provides the patient fields the policies require.
    Files without follow-up patients are skipped.

    Args:
        policies (PolicyStack): The run's policies.
        agent_data_path (str): An agent data JSON file, or a directory of them.
    """
    if not policies.required_patient_fields:
        return
    
    files = [agent_data_path] if agent_data_path.endswith('.json') else get_files(agent_data_path, ext='json')
    for path in files:
        records = followup_patient_records(json_load(path))
        if records:
            policies.check_patient_fields(available_patient_fields(records.values()), where=path)



def run_policies(metadata: Optional[dict] = None, where: str = 'run') -> list:
    """
    The policy list a simulation run recorded under ``_metadata.policies`` (see `PolicyStack.describe`).

    Args:
        metadata (Optional[dict], optional): The run's ``_metadata`` block. Defaults to None.
        where (str, optional): Label of the run (e.g. its result file), prefixed to error messages. Defaults to 'run'.

    Returns:
        list: One ``{'id', 'name', 'source', 'source_sha256', 'api_version', 'hooks', 'params'}`` per policy, in priority order.

    Raises:
        PolicyError: If the result is pre-plugin (no ``policies``, or ``policy`` still present) or recorded no policies.
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



def run_policy_summary(metadata: Optional[dict] = None, where: str = 'run') -> dict:
    """
    The negotiation mode and urgent capacity protection of a run, read from its ``_metadata.policies``.
    Review = first `result_gain` policy, else first `review` policy; without one the mode is ``'patient-side'``.

    Args:
        metadata (Optional[dict], optional): The run's ``_metadata`` block. Defaults to None.
        where (str, optional): Label of the run (e.g. its result file), prefixed to error messages. Defaults to 'run'.

    Returns:
        dict: ``{'mode', 'review', 'capacity_protection', 'policies'}``; ``mode`` is None for a custom review policy without one, ``review``/``capacity_protection`` (params) None when absent.
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
