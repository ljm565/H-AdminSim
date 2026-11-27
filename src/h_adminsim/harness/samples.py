import os
import json
import time
import random
import hashlib
from datetime import timedelta
from typing import Optional

from h_adminsim import __version__
from h_adminsim.registry.harness import (
    HARNESS_SAMPLES_FILE,
    HARNESS_SAMPLES_VERSION,
    HARNESS_SAMPLE_REVIEW_PER_KIND,
    HARNESS_SAMPLE_REVIEW_MAX,
    HARNESS_SAMPLE_TIME_BUDGET_S,
    HARNESS_SAMPLE_SHIFT_LEAD_HOURS,
    HARNESS_SAMPLE_SCHEDULE_BUDGET_S,
    HARNESS_SAMPLE_BUSY_LOADS,
)
from h_adminsim.registry.policy import POLICY_BUILTIN_METRIC_IDS
from h_adminsim.tools.scheduling_rule import SchedulingRule
from h_adminsim.environment.hospital import HospitalEnvironment
from h_adminsim.tools.negotiation_metrics import NegotiationMetrics
from h_adminsim.policy.context import RestrictContext, ReviewContext
from h_adminsim.simulation.opfu_scheduling_simulation import OPFUSchedulingSimulation
from h_adminsim.utils import log
from h_adminsim.utils.filesys_utils import get_files, json_load
from h_adminsim.utils.policy_utils import followup_patient_records
from h_adminsim.utils.common_utils import encode_non_finite, decode_non_finite, str_to_datetime

# Scheduling objective the staff books for each preference (`indifferent` patients get the earliest results)
_SCHEDULING_MODE = {'visit_min': 'visit_min', 'stay_min': 'stay_min', 'throughput_max': 'throughput_max'}
_MEMO: dict = {}       # digest -> samples built in this process
_DIGESTS: dict = {}    # (file, mtime, size) stamps -> digest



def _agent_data_files(agent_data_path: str) -> list[str]:
    """
    The agent data JSON files of a file or directory path, sorted.

    Args:
        agent_data_path (str): An agent data JSON file, or a directory of them.

    Returns:
        list[str]: The files.
    """
    return [agent_data_path] if agent_data_path.endswith('.json') else sorted(get_files(agent_data_path, ext='json'))



def agent_data_digest(agent_data_path: str) -> str:
    """
    Fingerprint of agent data, the package version and the sample format version, which together determine the
    generated samples.

    Args:
        agent_data_path (str): An agent data JSON file, or a directory of them.

    Returns:
        str: A sha256 hex digest.
    """
    files = _agent_data_files(agent_data_path)
    stamp = tuple((p, os.path.getmtime(p), os.path.getsize(p)) for p in files)   # unchanged files are not hashed again
    if stamp not in _DIGESTS:
        digest = hashlib.sha256(f"{__version__}/{HARNESS_SAMPLES_VERSION}".encode())
        for path in files:
            digest.update(os.path.basename(path).encode())
            with open(path, 'rb') as f:
                digest.update(f.read())
        _DIGESTS[stamp] = digest.hexdigest()
    return _DIGESTS[stamp]



def _gain_kind(gpc: float) -> str:
    """
    Kind of the hospital's gain from a patient switching to ``throughput_max``.

    Args:
        gpc (float): The patient's gain per concession.

    Returns:
        str: ``'zero'``, ``'finite'`` or ``'inf'`` (a free win).
    """
    if not gpc:
        return 'zero'
    return 'inf' if gpc == float('inf') else 'finite'



def _review_context(data: dict, env: HospitalEnvironment, gt: dict, record: dict) -> Optional[ReviewContext]:
    """
    The review context of one follow-up patient, as the simulator builds it when the staff books the patient's own
    preference with the scheduling tool: the preferred schedule, the ``throughput_max`` counterfactual and the metrics.

    Args:
        data (dict): The hospital's agent data.
        env (HospitalEnvironment): The hospital environment.
        gt (dict): The patient's ground-truth entry.
        record (dict): The patient record (`ctx.patient`).

    Returns:
        Optional[ReviewContext]: The context, or None if the tests cannot all be scheduled.
    """
    preference = (gt.get('preference') or ['throughput_max'])[0]
    codes = [t['code'] for t in gt.get('required_tests') or []]
    rule = SchedulingRule(data['metadata'], data['department'], env)
    devices = env.get_test_device_schedule(test_code=codes, test_device_information=data['test'])
    result = rule.schedule_tests(_SCHEDULING_MODE.get(preference, 'throughput_max'), devices, codes,
                                 time_budget_s=HARNESS_SAMPLE_SCHEDULE_BUDGET_S, unavailable=gt.get('unavailable'))
    if result['status'] != 'ok' or result['unscheduled']:
        return None
    preferred = OPFUSchedulingSimulation.postprocessing('tool_calling', {**result, 'fu_schedule': {'doctor': [], 'schedule': []}})
    metrics = NegotiationMetrics(preference=preference, achieved_schedule=preferred, filtered_test_device_information=devices,
                                 rule=rule, environment=env, unavailable=gt.get('unavailable')).to_dict()
    return ReviewContext(preference=preference,
                         metrics={k: metrics[k] for k in POLICY_BUILTIN_METRIC_IDS | {'preference'}},
                         patient=record,
                         schedules={'preferred': metrics['_P'], 'hospital': metrics['_T']})



def _restrict_contexts(data: dict, env: HospitalEnvironment, record: dict) -> list[RestrictContext]:
    """
    Snapshots of one hospital's test capacity for `restrict` hooks:
        - the simulation start (the fixed and pre-filled bookings only);
        - later, so that the first open date opens `HARNESS_SAMPLE_SHIFT_LEAD_HOURS` from now (the start has no date
          opening within a day);
        - the later one on a busy day: each open (test, date) booked to at least a share of its capacity cycling through
          `HARNESS_SAMPLE_BUSY_LOADS` (a synthetic load, since the samples do not run the simulation that fills dates).

    Args:
        data (dict): The hospital's agent data.
        env (HospitalEnvironment): The hospital environment at the simulation start.
        record (dict): A patient record for `ctx.patient` (the smoke test swaps in other patients).

    Returns:
        list[RestrictContext]: The snapshots.
    """
    start = RestrictContext(current_time=env.current_time, start_hour=data['metadata']['time']['start_hour'],
                            utc_offset=env._utc_offset, test_capacity=env.test_capacity,
                            test_duration=env.test_duration, booked_hours=env.booked_test_hours(), patient=record)
    open_pairs = sorted((c, d) for (c, d), h in start.test_capacity.items() if h > 0)
    if not open_pairs:
        return [start]
    first = min(start.lead_hours(d) for _, d in open_pairs)
    later_time = str_to_datetime(start.current_time) + timedelta(hours=first - HARNESS_SAMPLE_SHIFT_LEAD_HOURS)
    later = {**start.to_dict(), 'current_time': later_time.isoformat(timespec='seconds')}
    busy_hours = dict(start.booked_hours)
    for i, pair in enumerate(open_pairs):
        load = HARNESS_SAMPLE_BUSY_LOADS[i % len(HARNESS_SAMPLE_BUSY_LOADS)]
        busy_hours[pair] = max(busy_hours.get(pair, 0.0), load * start.test_capacity[pair])
    busy = {**later, 'booked_hours': [[c, d, h] for (c, d), h in busy_hours.items()]}
    return [start, RestrictContext.from_dict(later), RestrictContext.from_dict(busy)]



def build_sample_contexts(agent_data_path: str,
                          per_kind: int = HARNESS_SAMPLE_REVIEW_PER_KIND,
                          max_review: int = HARNESS_SAMPLE_REVIEW_MAX,
                          time_budget_s: float = HARNESS_SAMPLE_TIME_BUDGET_S,
                          seed: int = 0) -> tuple[list[ReviewContext], list[RestrictContext]]:
    """
    Smoke-test inputs built from agent data with the simulator's own scheduler and metrics (no LLM).

    Review contexts are spread over (preference, urgency, gain kind): follow-up patients are taken in a seeded random
    order, one (preference, urgency) group at a time in turn, and kept while their kind has fewer than `per_kind`
    samples; a group whose three gain kinds are all filled is skipped. It stops at `max_review` samples, when every
    group is filled or exhausted, or at the time budget.
    Restrict contexts are capacity snapshots of every hospital (see `_restrict_contexts`).

    Args:
        agent_data_path (str): An agent data JSON file, or a directory of them.
        per_kind (int, optional): Review samples per (preference, urgency, gain kind). Defaults to
                                  `HARNESS_SAMPLE_REVIEW_PER_KIND`.
        max_review (int, optional): Most review samples. Defaults to `HARNESS_SAMPLE_REVIEW_MAX`.
        time_budget_s (float, optional): Wall-clock budget for computing review samples. Defaults to
                                         `HARNESS_SAMPLE_TIME_BUDGET_S`.
        seed (int, optional): Seed of the patient order. Defaults to 0.

    Returns:
        tuple[list[ReviewContext], list[RestrictContext]]: The review and restrict contexts.
    """
    rng = random.Random(seed)
    hospitals = []
    for path in _agent_data_files(agent_data_path):
        data = json_load(path)
        records = followup_patient_records(data)
        if not records:
            continue
        env = HospitalEnvironment(data)
        patients = [gt for gt, _ in data.get('agent_data') or [] if gt.get('visit_type') == 'follow_up_visit']
        rng.shuffle(patients)
        hospitals.append((data, env, records, patients))

    restrict = [ctx for data, env, records, _ in hospitals
                for ctx in _restrict_contexts(data, env, rng.choice(list(records.values())))]

    # Patients grouped by (preference, urgency), in seeded random order within each group
    groups = {}
    for hospital in hospitals:
        for gt in hospital[3]:
            record = hospital[2].get(gt.get('patient'))
            if record:
                key = ((gt.get('preference') or ['throughput_max'])[0], record.get('urgency'))
                groups.setdefault(key, []).append((hospital, gt, record))
    for members in groups.values():
        rng.shuffle(members)

    review, kinds, start = [], {}, time.time()
    while groups and len(review) < max_review and time.time() - start < time_budget_s:
        for key in sorted(groups, key=str):     # one patient per group per round
            members = groups[key]
            if not members or all(kinds.get((*key, g), 0) >= per_kind for g in ('zero', 'finite', 'inf')):
                del groups[key]
                continue
            (data, env, _, _), gt, record = members.pop()
            try:
                ctx = _review_context(data, env, gt, record)
            except Exception as e:
                log(f"Skipped a sample patient ({type(e).__name__}: {e})", level='warning')
                continue
            kind = (*key, _gain_kind(ctx.metrics['gpc'])) if ctx else None
            if ctx and kinds.get(kind, 0) < per_kind:
                kinds[kind] = kinds.get(kind, 0) + 1
                review.append(ctx)
            if len(review) >= max_review or time.time() - start >= time_budget_s:
                break
    return review, restrict



def sample_contexts(agent_data_path: str, cache_dir: Optional[str] = None) -> tuple[list[ReviewContext], list[RestrictContext]]:
    """
    The smoke-test inputs for agent data: built once per data (`build_sample_contexts`), then reused from memory and, with
    `cache_dir`, from ``<cache_dir>/samples.json`` (rebuilt when the data or the package version changes).

    Args:
        agent_data_path (str): An agent data JSON file, or a directory of them.
        cache_dir (Optional[str], optional): Directory of the cache file (e.g. the policy directory). Defaults to None.

    Returns:
        tuple[list[ReviewContext], list[RestrictContext]]: The review and restrict contexts.
    """
    digest = agent_data_digest(agent_data_path)
    cache = os.path.join(cache_dir, HARNESS_SAMPLES_FILE) if cache_dir else None
    cached = None
    if cache and os.path.isfile(cache):
        with open(cache) as f:
            cached = json.load(f)
        if cached.get('digest') != digest:
            cached = None           # built from other data or another package version
    stored = _MEMO.get(digest) or cached
    if stored is None:
        log(f"Building smoke-test samples from {agent_data_path} ...")
        review, restrict = build_sample_contexts(agent_data_path)
        stored = encode_non_finite({'digest': digest, 'source': os.path.abspath(agent_data_path),
                                    'review': [c.to_dict() for c in review], 'restrict': [c.to_dict() for c in restrict]})
    if cache and cached is None:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache, 'w') as f:
            json.dump(stored, f, indent=4, ensure_ascii=False, allow_nan=False)   # readable; inf already stored as 'inf'
    _MEMO[digest] = stored
    return ([ReviewContext.from_dict({**c, 'metrics': decode_non_finite(c['metrics'])}) for c in stored['review']],
            [RestrictContext.from_dict(c) for c in stored['restrict']])
