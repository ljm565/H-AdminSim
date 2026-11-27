import os
import ast
import json
import inspect
import traceback
from collections import Counter
from dataclasses import replace
from typing import Callable, Iterable, Optional, Union

import h_adminsim
from h_adminsim.registry.errors import PolicyError, PolicyCompositionError, PolicyRequirementError
from h_adminsim.registry.harness import HARNESS_MAX_PATIENT_VARIANTS
from h_adminsim.registry.policy import POLICY_HOOKS, POLICY_FILE_PREFIX
from h_adminsim.policy.base import Policy
from h_adminsim.policy.builder import build_policy, load_policy_config
from h_adminsim.policy.context import ReviewContext
from h_adminsim.policy.stack import PolicyStack
from h_adminsim.harness.report import Report, plain_text, summarize_values
from h_adminsim.harness.samples import sample_contexts, agent_data_digest
from h_adminsim.utils.filesys_utils import get_files, json_load
from h_adminsim.utils.policy_utils import followup_patient_records, available_patient_fields

_PACKAGE_DIR = os.path.dirname(os.path.abspath(h_adminsim.__file__))

# Modules whose use in a hook makes it non-deterministic
_NONDETERMINISTIC_MODULES = ('random', 'secrets', 'uuid')



def load_patient_records(agent_data_path: str) -> list[dict]:
    """
    Patient records (as `ctx.patient` holds them) of every follow-up patient in the agent data.

    Args:
        agent_data_path (str): An agent data JSON file, or a directory of them.

    Returns:
        list[dict]: The records, one per patient.
    """
    files = [agent_data_path] if agent_data_path.endswith('.json') else get_files(agent_data_path, ext='json')
    records = {}
    for path in files:
        records.update(followup_patient_records(json_load(path)))
    return list(records.values())



def _user_frame(exc: BaseException, files: Optional[Iterable[str]] = None) -> Optional[str]:
    """
    Location of the innermost traceback frame in user code, as ``file.py:line `code```.

    Args:
        exc (BaseException): The exception.
        files (Optional[Iterable[str]], optional): Absolute paths counted as user code; None counts every file outside
                                                   the h_adminsim package and the import machinery. Defaults to None.

    Returns:
        Optional[str]: The location, or None if no frame is in user code.
    """
    files = {os.path.abspath(f) for f in files} if files is not None else None
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        path = os.path.abspath(frame.filename)
        if files is not None and path not in files:
            continue
        if files is None and (path.startswith(_PACKAGE_DIR) or frame.filename.startswith('<')):
            continue
        code = f" `{frame.line.strip()}`" if frame.line else ''
        return f"{os.path.basename(frame.filename)}:{frame.lineno}{code}"
    return None



def _parse(path: str, report: Report) -> Optional[ast.Module]:
    """
    Parse a policy file, reporting a missing file or a syntax error.

    Args:
        path (str): The Python file.
        report (Report): Report the issues are added to.

    Returns:
        Optional[ast.Module]: The syntax tree, or None on failure.
    """
    if not os.path.isfile(path):
        report.error('file_not_found', f"Policy file not found: {path}")
        return None
    with open(path) as f:
        source = f.read()
    try:
        return ast.parse(source, filename=path)
    except SyntaxError as e:
        report.error('syntax_error', f"{e.msg}: {(e.text or '').strip()}", f"{os.path.basename(path)}:{e.lineno}")
        return None



def _class_names(tree: ast.Module, base: str) -> list[str]:
    """
    Top-level classes of a file that subclass `base` directly.

    Args:
        tree (ast.Module): The file's syntax tree.
        base (str): The base class name (``'Policy'`` or ``'Metric'``).

    Returns:
        list[str]: The class names, in file order.
    """
    names = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [b.id if isinstance(b, ast.Name) else getattr(b, 'attr', None) for b in node.bases]
            if base in bases:
                names.append(node.name)
    return names



def _static_checks(tree: ast.Module, path: str, report: Report) -> None:
    """
    Warn about code that breaks the purity contract: hooks assigning to ``self`` or using ``global``, and imports of
    non-deterministic modules.

    Args:
        tree (ast.Module): The file's syntax tree.
        path (str): The file, for locations.
        report (Report): Report the warnings are added to.
    """
    name = os.path.basename(path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            used = sorted({m.split('.')[0] for m in modules} & set(_NONDETERMINISTIC_MODULES))
            if used:
                report.warning('nondeterministic', f"Imports {used}; hooks must be pure functions of `ctx` (the same "
                                                   f"input must give the same output).", f"{name}:{node.lineno}")
        if not isinstance(node, ast.ClassDef):
            continue
        for fn in node.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name in POLICY_HOOKS + ('compute',)):
                continue
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Global):
                    report.warning('stateful_hook', f"`{node.name}.{fn.name}` uses `global`; hooks must not keep state "
                                                    f"across patients.", f"{name}:{sub.lineno}")
                targets = sub.targets if isinstance(sub, ast.Assign) else \
                    [sub.target] if isinstance(sub, (ast.AugAssign, ast.AnnAssign)) else []
                for t in targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self':
                        report.warning('stateful_hook',
                                       f"`{node.name}.{fn.name}` assigns `self.{t.attr}`; hooks must not keep state across "
                                       f"patients (patients run in separate worker processes and runs resume). Compute it "
                                       f"from `ctx` instead.", f"{name}:{sub.lineno}")



def _source_files(stack: PolicyStack) -> dict:
    """
    Source file of every policy and metric of a stack.

    Args:
        stack (PolicyStack): The policies.

    Returns:
        dict: ``{absolute path: id}``.
    """
    files = {}
    for obj in stack.policies + stack.metrics:
        if obj.source and obj.source.startswith(POLICY_FILE_PREFIX):
            path = obj.source[len(POLICY_FILE_PREFIX):].rpartition(':')[0]
        else:
            path = inspect.getsourcefile(type(obj))
        if path:
            files.setdefault(os.path.abspath(path), obj.id)
    return files



def _check_patient_fields(stack: PolicyStack, report: Report, records: list, label: str) -> bool:
    """
    Check that the patient records hold every patient field the stack requires.

    Args:
        stack (PolicyStack): The policies.
        report (Report): Report a missing field is added to.
        records (list): Patient records.
        label (str): Name of the records' origin, for the message.

    Returns:
        bool: True if every required field is present.
    """
    try:
        stack.check_patient_fields(available_patient_fields(records), where=label)
        return True
    except PolicyRequirementError as e:
        report.error('missing_patient_fields', e)
        return False



def _with_patient_variants(contexts: list, pool: list, fields: Iterable[str],
                           cap: int = HARNESS_MAX_PATIENT_VARIANTS) -> list[tuple]:
    """
    Every context once per distinct combination of the patient fields the policies read (one patient per combination,
    at most `cap`), so each patient condition is tested against every metric scenario instead of only the patient a
    sample happened to have. Without such fields the contexts are kept as they are.

    Args:
        contexts (list): Review or restrict contexts.
        pool (list): Patient records to pick from.
        fields (Iterable[str]): The patient fields the policies read (``requires``).
        cap (int, optional): Maximum number of patients per context. Defaults to `HARNESS_MAX_PATIENT_VARIANTS`.

    Returns:
        list[tuple]: ``(context, tag)`` pairs; the tag shows the patient's field values (empty without fields).
    """
    fields = sorted(fields)
    if not fields:
        return [(ctx, '') for ctx in contexts]
    variants = {}
    for record in pool:
        key = tuple(json.dumps(record.get(f), sort_keys=True, default=str) for f in fields)
        if key not in variants:
            variants[key] = record
            if len(variants) >= cap:
                break
    tags = {key: ', '.join(f"{f}={v if len(v) <= 30 else v[:27] + '...'}" for f, v in zip(fields, key)) for key in variants}
    return [(replace(ctx, patient=record), tags[key]) for ctx in contexts for key, record in variants.items()]



def _never_active_message(policy_id: str, hook: str, samples: int) -> str:
    """
    Warning text for a policy whose hook did nothing on any sample.

    Args:
        policy_id (str): The policy id.
        hook (str): ``'restrict'`` or ``'review'``.
        samples (int): Number of samples.

    Returns:
        str: The message.
    """
    what = (f"`{policy_id}`.restrict blocked no (test_code, date) pair in any of the {samples} samples" if hook == 'restrict'
            else f"`{policy_id}`.review abstained (returned None) in all {samples} samples")
    return (f"{what}. Check its conditions (comparisons, field values, units, date handling); if it targets situations "
            f"rarer than the samples cover, say so and confirm it with a simulation run.")



def _decides(policy: Policy, ctx: ReviewContext) -> bool:
    """
    Whether a `review` policy, asked alone, decides (does not abstain on) a context. Errors count as abstaining, since
    the stack run reports them.

    Args:
        policy (Policy): The `review` policy.
        ctx (ReviewContext): The context, with every metric the stack computed.

    Returns:
        bool: True if the policy returns a decision.
    """
    try:
        return policy.review(ctx.for_policy(policy)) is not None
    except Exception:
        return False



def _report_review_conflicts(ids: list, would: Counter, decided: Counter, both: Counter, samples: int, report: Report) -> None:
    """
    Warn about `review` policies competing for the same patients: a lower-priority policy hidden by higher ones
    (``review_shadowed``), or two policies that both decide some patients (``review_overlap``). Each warning lists the
    options the user can pick from.

    Args:
        ids (list): The `review` policy ids in priority order.
        would (Counter): Samples each policy decides when asked alone.
        decided (Counter): Samples each policy actually decided in the stack.
        both (Counter): ``{(higher id, lower id): samples both would decide}``.
        samples (int): Number of review samples.
        report (Report): Report the warnings are added to.
    """
    shadowed = set()
    for b in ids[1:]:
        if would[b] and not decided[b]:
            shadowed.add(b)
            higher = [a for a in ids[:ids.index(b)] if both[(a, b)]]
            report.warning('review_shadowed',
                           f"`{b}` never decides: on all {would[b]} of {samples} samples where it would decide, the "
                           f"higher-priority {', '.join(f'`{a}`' for a in higher)} {'decides' if len(higher) == 1 else 'decide'} "
                           f"first. Options: (1) put `{b}` "
                           f"before them, (2) make the higher-priority policy return None for those patients, "
                           f"(3) drop `{b}`.")
    for (a, b), n in both.items():
        if b in shadowed:
            continue
        report.warning('review_overlap',
                       f"`{a}` and `{b}` would both decide {n} of {samples} samples; `{a}` decides them (higher "
                       f"priority) and `{b}` only the rest. Options: (1) keep this order, (2) put `{b}` first, "
                       f"(3) narrow one of them so it returns None for those patients.")



def _smoke(stack: PolicyStack, report: Report, review_ctxs: list, restrict_ctxs: list, records: list) -> None:
    """
    Run the stack's hooks on the sample contexts (twice, to catch non-determinism) and report exceptions, invalid
    outputs, conflicts between policies and policies that never act; the outcomes go to ``report.info['smoke']``.

    Args:
        stack (PolicyStack): The policies.
        report (Report): Report the findings are added to.
        review_ctxs (list): Review sample contexts.
        restrict_ctxs (list): Restrict sample contexts.
        records (list): Agent-data patient records; each context is run with patients covering the values of the
                        patient fields the policies read (see `_with_patient_variants`).
    """
    fields = stack.required_patient_fields
    review_ctxs, restrict_ctxs = (_with_patient_variants(review_ctxs, records, fields),
                                  _with_patient_variants(restrict_ctxs, records, fields))
    files = _source_files(stack)
    problems = {}   # (level, code, message, where) -> sample labels

    def note(level: str, code: str, message: str, sample: str, where: Optional[str] = None) -> None:
        problems.setdefault((level, code, plain_text(message), where), []).append(sample)

    def run(call: Callable, sample: str) -> tuple:
        try:
            return True, call()
        except PolicyRequirementError as e:
            note('error', 'requirement', e, sample, _user_frame(e, files))
        except PolicyError as e:
            note('error', 'bad_output', e, sample, _user_frame(e, files))
        except Exception as e:
            note('error', 'hook_raised', f"{type(e).__name__}: {e}", sample, _user_frame(e, files))
        return False, None

    smoke = report.info.setdefault('smoke', {})
    if stack.restrictors:
        blocked = {p.id: [] for p in stack.restrictors}
        shared = Counter()      # (a, b) -> samples where both block some same pair
        shared_pairs = Counter()
        for i, (ctx, tag) in enumerate(restrict_ctxs):
            label = f"restrict sample {i}" + (f" ({tag})" if tag else '')
            ok, first = run(lambda: stack.restrict_by_policy(ctx), label)
            if not ok:
                continue
            ok, second = run(lambda: stack.restrict_by_policy(ctx), label)
            if ok and first != second:
                note('warning', 'nondeterministic', "`restrict` returned different pairs for the same input; hooks must be "
                                                    "pure functions of `ctx` (no randomness, no state).", label)
            for pid, pairs in first.items():
                blocked[pid].append(len(pairs))
                unknown = sorted(pairs - set(ctx.test_capacity))
                if unknown:
                    note('warning', 'unknown_pairs', f"`{pid}`.restrict blocked pairs that are not (test_code, date) keys "
                                                     f"of `ctx.test_capacity`, e.g. {unknown[:3]}; check the test code and "
                                                     f"the date format ('YYYY-MM-DD').", label)
            ids = list(first)
            for x, a in enumerate(ids):
                for b in ids[x + 1:]:
                    common = first[a] & first[b]
                    if common:
                        shared[(a, b)] += 1
                        shared_pairs[(a, b)] += len(common)
        smoke['restrict'] = {'samples': len(restrict_ctxs), 'blocked_pairs_per_sample': blocked}
        if not any(level == 'error' for level, *_ in problems):     # a failing hook says nothing about its conditions
            for pid, counts in blocked.items():
                if counts and not any(counts):
                    report.warning('never_active', _never_active_message(pid, 'restrict', len(restrict_ctxs)))
        for (a, b), n in shared.items():
            report.warning('restrict_overlap',
                           f"`{a}` and `{b}` block the same (test_code, date) pairs on {n} of {len(restrict_ctxs)} samples "
                           f"({shared_pairs[(a, b)]} pairs). Blocks are combined, so both apply; check that neither was "
                           f"meant to allow what the other blocks.")

    if stack.reviewers or stack.metrics:
        outcomes, values = Counter(), {m.id: [] for m in stack.metrics}
        would, decided, both = Counter(), Counter(), Counter()   # per policy / per (higher, lower) pair
        for i, (ctx, tag) in enumerate(review_ctxs):
            label = f"review sample {i}" + (f" ({tag})" if tag else '')
            ok, custom = run(lambda: stack.compute_metrics(ctx), label)
            if not ok:
                continue
            for key, value in custom.items():
                values[key].append(value)
            full = replace(ctx, metrics={**ctx.metrics.to_dict(), **custom}) if custom else ctx
            ok, first = run(lambda: stack.review(full), label)
            if not ok:
                continue
            ok, second = run(lambda: stack.review(full), label)
            if ok and first != second:
                note('warning', 'nondeterministic', "`review` returned different decisions for the same input; hooks must "
                                                    "be pure functions of `ctx` (no randomness, no state).", label)
            outcomes[f"{first.decided_by}: {first.action}" if first.decided_by else 'keep (every policy abstained)'] += 1
            decided[first.decided_by] += 1
            wants = [p.id for p in stack.reviewers if _decides(p, full)]   # who would decide if asked alone
            would.update(wants)
            both.update((a, b) for x, a in enumerate(wants) for b in wants[x + 1:])
        smoke['review'] = {'samples': len(review_ctxs), 'outcomes': dict(outcomes)}
        if values:
            smoke['metric_values'] = {key: summarize_values(v) for key, v in values.items()}
            if not any(level == 'error' for level, *_ in problems):
                for key, summary in smoke['metric_values'].items():
                    if summary['samples'] > 1 and summary['distinct'] == 1:
                        report.warning('constant_metric',
                                       f"Metric `{key}` has the same value ({values[key][0]!r}) on all {summary['samples']} "
                                       f"samples, so it cannot tell patients apart. Check what it reads and how it "
                                       f"computes the value.")
        if len(stack.reviewers) > 1:
            smoke['review']['would_decide_alone'] = {p.id: would[p.id] for p in stack.reviewers}
            _report_review_conflicts([p.id for p in stack.reviewers], would, decided, both, len(review_ctxs), report)
        if not any(level == 'error' for level, *_ in problems):     # a failing hook says nothing about its conditions
            for p in stack.reviewers:
                if not would[p.id]:
                    report.warning('never_active', _never_active_message(p.id, 'review', len(review_ctxs)))

    for (level, code, message, where), samples in problems.items():
        more = f", ... ({len(samples)} in total)" if len(samples) > 3 else ''
        text = f"{message} [{', '.join(samples[:3])}{more}]"
        (report.error if level == 'error' else report.warning)(code, text, where or samples[0])



def _smoke_with_data(stack: PolicyStack, report: Report, agent_data_path: str, sample_dir: Optional[str] = None) -> None:
    """
    Check the required patient fields against the agent data, build (or reuse) the smoke-test samples from it, and run
    the smoke test.

    Args:
        stack (PolicyStack): The policies.
        report (Report): Report the findings are added to.
        agent_data_path (str): An agent data JSON file or directory.
        sample_dir (Optional[str], optional): Directory caching the samples built from the data (see `sample_contexts`).
                                              Defaults to None.
    """
    try:
        records = load_patient_records(agent_data_path)
    except Exception as e:
        report.error('agent_data_unreadable', f"Could not read agent data {agent_data_path}: {type(e).__name__}: {e}")
        return
    if not records:
        report.error('no_followup_patients', f"{agent_data_path} has no follow-up patients to test the policies on.")
        return
    report.info['agent_data_patients'] = len(records)
    if not _check_patient_fields(stack, report, records, agent_data_path):
        return

    try:
        review_ctxs, restrict_ctxs = sample_contexts(agent_data_path, sample_dir)
    except Exception as e:
        report.error('samples_failed', f"Could not build smoke-test samples from {agent_data_path}: {type(e).__name__}: {e}")
        return
    report.info['samples'] = {'review': len(review_ctxs), 'restrict': len(restrict_ctxs),
                              'digest': agent_data_digest(agent_data_path)[:12]}
    _smoke(stack, report, review_ctxs, restrict_ctxs, records)



def validate_policy(path: str,
                    class_name: Optional[str] = None,
                    params: Optional[dict] = None,
                    *,
                    agent_data_path: str,
                    sample_dir: Optional[str] = None) -> Report:
    """
    Check one policy file: syntax, the policy class contract, the parameters, purity warnings, and a smoke test of its
    hooks on sample contexts. Never raises on a broken policy.

    Args:
        path (str): The policy's Python file.
        class_name (Optional[str], optional): The policy class; None picks the file's only `Policy` subclass.
                                              Defaults to None.
        params (Optional[dict], optional): Parameters to build the policy with (its config ``params``). Defaults to None.
        agent_data_path (str): Agent data the smoke-test samples are built from (it must hold the required patient
                               fields).
        sample_dir (Optional[str], optional): Directory caching the samples (see `sample_contexts`). Defaults to None.

    Returns:
        Report: The findings; ``info`` holds the policy's id, hooks, requires, params and smoke-test outcomes.
    """
    report = Report(info={'path': os.path.abspath(path)})
    tree = _parse(path, report)
    if tree is None:
        return report

    if class_name is None:
        candidates = _class_names(tree, 'Policy')
        metrics = _class_names(tree, 'Metric')
        if not candidates and metrics:
            report.error('metric_needs_config', f"{os.path.basename(path)} defines the metric(s) {metrics} and no policy. "
                                                f"A metric runs only from a policy config that lists it under `metrics:`; "
                                                f"save or validate it with that config (usually together with the policy "
                                                f"that reads it).")
            return report
        if len(candidates) != 1:
            found = f"found {candidates}" if candidates else 'found none'
            report.error('class_not_found', f"Expected exactly one class subclassing `Policy` in {os.path.basename(path)} "
                                            f"({found}); pass `class_name`.")
            return report
        class_name = candidates[0]
    _static_checks(tree, path, report)

    try:
        policy = build_policy(f"{POLICY_FILE_PREFIX}{os.path.abspath(path)}:{class_name}", params)
    except Exception as e:
        cause = e.__cause__ if isinstance(e, PolicyError) and e.__cause__ else e
        message = plain_text(e) if isinstance(e, PolicyError) else f"{type(e).__name__}: {e}"
        code = 'import_failed' if 'Failed to import' in message else \
            'bad_params' if 'Invalid parameters' in message else 'build_failed'
        report.error(code, message, _user_frame(cause, [path]))
        return report

    report.info.update({'class': class_name, 'id': policy.id, 'name': policy.name, 'hooks': list(policy.hooks()),
                        'requires': list(policy.requires), 'params': policy.params()})
    try:
        stack = PolicyStack([policy])
    except PolicyError as e:
        report.warning('smoke_skipped', f"{plain_text(e)} Smoke-test it with `validate_config` on a config that lists "
                                        f"those metrics.")
        return report
    _smoke_with_data(stack, report, agent_data_path, sample_dir)
    return report



def validate_config(paths: Union[str, os.PathLike, list],
                    *,
                    agent_data_path: str,
                    sample_dir: Optional[str] = None) -> Report:
    """
    Check policy config files as a run would load them (in priority order), then smoke-test the combined policies and
    metrics on sample contexts. Never raises on a broken config or policy.

    Args:
        paths (Union[str, os.PathLike, list]): A policy config file, or a list of them in priority order.
        agent_data_path (str): Agent data the smoke-test samples are built from (it must hold the required patient
                               fields).
        sample_dir (Optional[str], optional): Directory caching the samples (see `sample_contexts`). Defaults to None.

    Returns:
        Report: The findings; ``info`` holds the loaded policies, metrics and evaluations and the smoke-test outcomes.
    """
    paths = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths or [])
    report = Report(info={'configs': [os.path.abspath(os.fspath(p)) for p in paths]})
    if not paths:
        report.error('no_config', "No policy config given.")
        return report

    try:
        stack = load_policy_config(paths)
    except PolicyError as e:
        code = 'composition' if isinstance(e, PolicyCompositionError) else \
            'requirement' if isinstance(e, PolicyRequirementError) else 'config_invalid'
        report.error(code, e, _user_frame(e.__cause__ or e))
        return report
    except Exception as e:   # e.g. a policy's __post_init__ raising something other than PolicyError
        report.error('build_failed', f"{type(e).__name__}: {e}", _user_frame(e))
        return report

    report.info.update({'policies': stack.describe(), 'metrics': stack.describe_metrics(),
                        'evaluations': stack.describe_evaluations()})
    for path in {p for p in _source_files(stack) if not p.startswith(_PACKAGE_DIR)}:
        tree = _parse(path, report)
        if tree is not None:
            _static_checks(tree, path, report)
    _smoke_with_data(stack, report, agent_data_path, sample_dir)
    return report
