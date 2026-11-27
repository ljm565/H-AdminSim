import re
from collections import Counter
from dataclasses import fields
from typing import Any, Optional

from h_adminsim.registry.harness import HARNESS_DESCRIBE_MAX_VALUES
from h_adminsim.registry.policy import POLICY_BUILTIN_METRIC_IDS
from h_adminsim.policy.context import RestrictContext, ReviewContext
from h_adminsim.harness.report import summarize_values
from h_adminsim.harness.samples import sample_contexts
from h_adminsim.harness.validate import load_patient_records

_SCALAR = (str, int, float, bool)
_ISO_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}')



def _empty(value: Any) -> bool:
    """
    Whether a value carries no information (None, empty string, list or dict).

    Args:
        value (Any): The value.

    Returns:
        bool: True if empty.
    """
    return value is None or value == '' or value == [] or value == {}



def _value_counts(values: list, max_values: int) -> dict:
    """
    Summary of scalar values: every value with its count when there are few distinct ones, else the number of distinct
    values, a few examples and (for numbers and ISO dates) the range.

    Args:
        values (list): Scalar values (empty ones excluded by the caller).
        max_values (int): Most distinct values listed one by one.

    Returns:
        dict: ``{'values': {value: count}}`` or ``{'distinct', 'examples'[, 'range']}``.
    """
    counts = Counter(values)
    if len(counts) <= max_values:
        return {'values': {str(k): n for k, n in counts.most_common()}}
    summary = {'distinct': len(counts), 'examples': [str(k) for k in list(counts)[:3]]}
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values) or \
            all(isinstance(v, str) and _ISO_DATE.match(v) for v in values):
        summary['range'] = [str(min(values)), str(max(values))]
    return summary



def _summarize(values: list, total: int, max_values: int) -> dict:
    """
    Summary of one field over the patient records.

    Args:
        values (list): The field's values in the records that have the field.
        total (int): Number of records.
        max_values (int): Most distinct values listed one by one.

    Returns:
        dict: ``{'present', 'type', ...}`` with value counts for scalars, item summaries for lists and per-key summaries
              (one level) for dicts.
    """
    filled = [v for v in values if not _empty(v)]
    summary = {'present': f"{len(filled)}/{total}",
               'type': sorted({type(v).__name__ for v in values if v is not None}) or ['null']}
    scalars = [v for v in filled if isinstance(v, _SCALAR)]
    lists = [v for v in filled if isinstance(v, list)]
    dicts = [v for v in filled if isinstance(v, dict)]
    if scalars:
        summary.update(_value_counts(scalars, max_values))
    if lists:
        items = [x for v in lists for x in v]
        summary['length'] = [min(len(v) for v in lists), max(len(v) for v in lists)]
        if any(isinstance(x, _SCALAR) for x in items):
            summary['items'] = _value_counts([x for x in items if isinstance(x, _SCALAR)], max_values)
        records = [x for x in items if isinstance(x, dict)]
        if records:
            summary['item_keys'] = sorted({k for x in records for k in x})
            summary['item_example'] = records[0]
    if dicts:
        summary['keys'] = {key: _summarize([d[key] for d in dicts if key in d], len(dicts), max_values)
                           for key in sorted({k for d in dicts for k in d})}
    return summary



def describe_patient_data(records: list[dict], max_values: int = HARNESS_DESCRIBE_MAX_VALUES) -> dict:
    """
    What patient information a set of follow-up patient records holds: for every field, how many patients have a
    non-empty value and which values occur.

    Args:
        records (list[dict]): Patient records, as `ctx.patient` holds them.
        max_values (int, optional): Most distinct values listed one by one per field. Defaults to
                                    `HARNESS_DESCRIBE_MAX_VALUES`.

    Returns:
        dict: ``{'patients', 'fields': {field: summary}}``.
    """
    keys = sorted({k for r in records for k in r})
    return {'patients': len(records),
            'fields': {k: _summarize([r[k] for r in records if k in r], len(records), max_values) for k in keys}}



def describe_data(agent_data_path: str, max_values: int = HARNESS_DESCRIBE_MAX_VALUES, sample_dir: Optional[str] = None) -> dict:
    """
    What a policy can rely on: the patient information in the agent data (field by field, see `describe_patient_data`),
    what the simulator always provides whatever the data (built-in metrics, schedules and hospital state), and the range
    of each built-in metric per preference in the smoke-test samples (to pick thresholds).

    Args:
        agent_data_path (str): Agent data JSON file or directory.
        max_values (int, optional): Most distinct values listed one by one per field. Defaults to
                                    `HARNESS_DESCRIBE_MAX_VALUES`.
        sample_dir (Optional[str], optional): Directory caching the smoke-test samples (see `sample_contexts`). Defaults
                                              to None.

    Returns:
        dict: ``{'patient_data': {'origin', 'patients', 'fields'}, 'always_available': {...},
              'metrics_in_samples': {preference: {metric: summary}}}``.
    """
    records = load_patient_records(agent_data_path)
    try:
        review, _ = sample_contexts(agent_data_path, sample_dir)
        ranges = None
    except Exception as e:     # e.g. data without the hospital part: the patient fields are still described
        review, ranges = [], f"unavailable: the samples could not be built ({type(e).__name__}: {e})"
    by_preference = {}
    for ctx in review:
        for metric in sorted(POLICY_BUILTIN_METRIC_IDS):
            by_preference.setdefault(ctx.preference, {}).setdefault(metric, []).append(ctx.metrics[metric])
    return {
        'patient_data': {'origin': f"agent data {agent_data_path}", **describe_patient_data(records, max_values)},
        'always_available': {
            'note': "Computed by the simulator for every patient, whatever the agent data holds.",
            'review_context': [f.name for f in fields(ReviewContext)],
            'builtin_metrics': sorted(POLICY_BUILTIN_METRIC_IDS),
            'schedules': "ctx.schedules['preferred'] / ['hospital']: test_schedule (code, device, date, start, end, "
                         "result_ready_at), test_visit_dates, idle_waiting_time, all_results_ready_at",
            'restrict_context': [f.name for f in fields(RestrictContext)] + ['lead_hours(date)'],
            'not_in_review_context': "device occupancy, free slots and the current time (only `tcl` summarizes test "
                                     "congestion there; restrict hooks see capacity and booked hours)",
        },
        'metrics_in_samples': ranges or {pref: {m: summarize_values(v) for m, v in metrics.items()}
                                         for pref, metrics in sorted(by_preference.items())},
    }
