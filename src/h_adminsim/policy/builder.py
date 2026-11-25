import os
import sys
import yaml
import shutil
import hashlib
import importlib.util
from typing import Any, Callable, Optional, Union

from h_adminsim.utils import colorstr
from h_adminsim.registry.errors import PolicyError
from h_adminsim.registry.policy import (
    POLICY_ENTRY_KEYS,
    POLICY_CONFIG_KEYS,
    POLICY_CONFIG_API_VERSION,
)
from h_adminsim.policy.base import Policy
from h_adminsim.policy.stack import PolicyStack
from h_adminsim.policy.metric import Metric
from h_adminsim.policy.evaluation import Evaluation
from h_adminsim.policy.registry import (
    get_policy_class,
    get_metric_class,
    get_evaluation_class,
    validate_policy_class,
    validate_metric_class,
    validate_evaluation_class,
)



def _unique_name(name: str, taken: set) -> str:
    """
    A file name not in `taken` (``name``, else ``<stem>_1<ext>``, ``<stem>_2<ext>``, ...), added to `taken`.

    Args:
        name (str): The preferred file name.
        taken (set): Names already used in the target directory; updated in place.

    Returns:
        str: The chosen file name.
    """
    stem, ext = os.path.splitext(name)
    candidate, n = name, 1
    while candidate in taken:
        candidate, n = f"{stem}_{n}{ext}", n + 1
    taken.add(candidate)
    return candidate



def _config_paths(paths: Optional[Union[str, os.PathLike, list]] = None) -> list:
    """
    Normalize a `policy_config` value into a list of absolute paths.

    Args:
        paths (Optional[Union[str, os.PathLike, list]], optional): One policy config path, or a list of paths in
                                                                   priority order (e.g. a YAML list). Defaults to None.

    Raises:
        PolicyError: If `paths` is neither a path nor an iterable of paths.

    Returns:
        list: Absolute paths in the given order (empty for None).
    """
    if paths is None:
        return []
    
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    
    try:
        paths = list(paths)
    except TypeError:
        raise PolicyError(colorstr("red", f"`policy_config` must be a path or a list of paths, got {type(paths).__name__}."))
    
    return [os.path.abspath(os.fspath(p)) for p in paths]



def _resolve_file_source(spec: str, 
                         base_dir: Optional[str] = None, 
                         kind: str = 'policy') -> tuple[str, str]:
    """
    Split a ``file:`` source spec into the absolute path of its Python file and the class name.

    Args:
        spec (str): The part after ``file:``, i.e. ``<path.py>:<ClassName>``. The path may be absolute or relative.
        base_dir (Optional[str], optional): Directory a relative path is resolved against. Defaults to None (the cwd).
        kind (str, optional): ``'policy'``, ``'metric'`` or ``'evaluation'``, for error messages. Defaults to 'policy'.

    Raises:
        PolicyError: If the spec is malformed or the file does not exist.

    Returns:
        tuple[str, str]: ``(absolute_path, class_name)``.
    """
    path, sep, class_name = spec.rpartition(':')
    if not sep or not path or not class_name:
        raise PolicyError(colorstr("red", f"File {kind} source must be `file:<path.py>:<ClassName>`, got `file:{spec}`."))
    
    if not os.path.isabs(path):
        path = os.path.join(base_dir or os.getcwd(), path)
    
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise PolicyError(colorstr("red", f"{kind.capitalize()} file not found: {path}"))
    
    return path, class_name



def _load_file_class(spec: str,
                     base_dir: Optional[str] = None,
                     validate: Callable = validate_policy_class,
                     kind: str = 'policy') -> tuple[type, str, str]:
    """
    Import the policy, metric or evaluation class named by a ``file:`` source spec (``path/to/policy.py:ClassName``).

    The file is imported as its own module, named after the file's sha256, so loading the same file twice reuses the
    module (also when one file defines, e.g., both a policy and an evaluation) and an edited file is imported afresh.

    Args:
        spec (str): The part after ``file:``, i.e. ``<path.py>:<ClassName>``. The path may be absolute or relative.
        base_dir (Optional[str], optional): Directory a relative path is resolved against. Defaults to None (the cwd).
        validate (Callable, optional): Checks the loaded class (`validate_policy_class`, `validate_metric_class` or
                                       `validate_evaluation_class`). Defaults to `validate_policy_class`.
        kind (str, optional): ``'policy'``, ``'metric'`` or ``'evaluation'``, for error messages. Defaults to 'policy'.

    Raises:
        PolicyError: If the file fails to import or does not define the class.

    Returns:
        tuple[type, str, str]: ``(cls, absolute_path, sha256_of_file)``.
    """
    path, class_name = _resolve_file_source(spec, base_dir, kind)

    with open(path, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    # One module per file content, so a re-load of the same file reuses it and an edited file gets a fresh module
    module_name = f"h_adminsim_user_code_{digest[:16]}"
    
    # Reuse the module if this file content was already imported
    module = sys.modules.get(module_name)

    # Otherwise import it
    if module is None:
        spec_obj = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[module_name] = module      # Register before executing, as `import` does (dataclasses look the module up)
        try:
            spec_obj.loader.exec_module(module)
        except Exception as e:
            del sys.modules[module_name]       # Drop the half-initialized module so a retry imports the file again
            raise PolicyError(colorstr("red", f"Failed to import {kind} file {path}: {type(e).__name__}: {e}")) from e

    cls = getattr(module, class_name, None)
    if cls is None:
        raise PolicyError(colorstr("red", f"{kind.capitalize()} file {path} defines no `{class_name}`."))
    validate(cls)
    
    return cls, path, digest



def build_policy(source: str,
                 params: Optional[dict] = None,
                 id: Optional[str] = None,
                 base_dir: Optional[str] = None) -> Policy:
    """
    Instantiate one policy.

    Args:
        source (str): ``builtin:<name>`` (or a bare registered ``<name>``) for a registered policy, or
            ``file:<path.py>:<ClassName>`` for a policy class defined in a Python file outside the package.
        params (Optional[dict], optional): The policy's dataclass fields. Defaults to None.
        id (Optional[str], optional): Instance id within the run. Defaults to the policy's `name`.
        base_dir (Optional[str], optional): Directory a relative ``file:`` path is resolved against. Defaults to the cwd.

    Raises:
        PolicyError: If `source` is not a non-empty string or `params` does not fit the policy.

    Returns:
        Policy: The policy, carrying `id`, `source` and (for file sources) `source_sha256`.
    """
    if not isinstance(source, str) or not source:
        raise PolicyError(colorstr("red", f"Policy `source` must be a non-empty string, got {source!r}."))

    digest = None
    if source.startswith('file:'):
        cls, path, digest = _load_file_class(source[len('file:'):], base_dir, validate=validate_policy_class, kind='policy')
        source = f"file:{path}:{cls.__name__}"
    else:
        name = source[len('builtin:'):] if source.startswith('builtin:') else source
        cls = get_policy_class(name)
        source = f"builtin:{name}"

    try:
        policy = cls(**dict(params or {}))
    except TypeError as e:
        raise PolicyError(colorstr("red", f"Invalid parameters for policy `{source}`: {e}")) from e
    
    policy.id = str(id) if id else cls.name
    policy.source = source
    policy.source_sha256 = digest
    
    return policy



def build_metric(source: str,
                 params: Optional[dict] = None,
                 id: Optional[str] = None,
                 base_dir: Optional[str] = None) -> Metric:
    """
    Instantiate one user-defined negotiation metric.

    Args:
        source (str): ``builtin:<name>`` (or a bare ``<name>``) for a metric registered with `register_metric`, or
            ``file:<path.py>:<ClassName>`` for a metric class defined in a Python file.
        params (Optional[dict], optional): The metric's dataclass fields. Defaults to None.
        id (Optional[str], optional): The metric's id, i.e. its key in ``ctx.metrics`` and the booking record.
                                      Defaults to the metric's `name`.
        base_dir (Optional[str], optional): Directory a relative ``file:`` path is resolved against. Defaults to the cwd.

    Raises:
        PolicyError: If `source` is not a non-empty string or `params` does not fit the metric.

    Returns:
        Metric: The metric, carrying `id`, `source` and (for file sources) `source_sha256`.
    """
    if not isinstance(source, str) or not source:
        raise PolicyError(colorstr("red", f"Metric `source` must be a non-empty string, got {source!r}."))

    digest = None
    if source.startswith('file:'):
        cls, path, digest = _load_file_class(source[len('file:'):], base_dir, validate=validate_metric_class, kind='metric')
        source = f"file:{path}:{cls.__name__}"
    else:
        name = source[len('builtin:'):] if source.startswith('builtin:') else source
        cls = get_metric_class(name)
        source = f"builtin:{name}"

    try:
        metric = cls(**dict(params or {}))
    except TypeError as e:
        raise PolicyError(colorstr("red", f"Invalid parameters for metric `{source}`: {e}")) from e
    
    metric.id = str(id) if id else cls.name
    metric.source = source
    metric.source_sha256 = digest
    
    return metric



def build_evaluation(source: str,
                     params: Optional[dict] = None,
                     id: Optional[str] = None,
                     base_dir: Optional[str] = None) -> Evaluation:
    """
    Instantiate one user-defined evaluation.

    Args:
        source (str): ``builtin:<name>`` (or a bare ``<name>``) for an evaluation registered with `register_evaluation`,
            or ``file:<path.py>:<ClassName>`` for an evaluation class defined in a Python file.
        params (Optional[dict], optional): The evaluation's dataclass fields. Defaults to None.
        id (Optional[str], optional): The evaluation's id, i.e. its key in the report. Defaults to its `name`.
        base_dir (Optional[str], optional): Directory a relative ``file:`` path is resolved against. Defaults to the cwd.

    Raises:
        PolicyError: If `source` is not a non-empty string or `params` does not fit the evaluation.

    Returns:
        Evaluation: The evaluation, carrying `id`, `source` and (for file sources) `source_sha256`.
    """
    if not isinstance(source, str) or not source:
        raise PolicyError(colorstr("red", f"Evaluation `source` must be a non-empty string, got {source!r}."))

    digest = None
    if source.startswith('file:'):
        cls, path, digest = _load_file_class(source[len('file:'):], base_dir, validate=validate_evaluation_class, kind='evaluation')
        source = f"file:{path}:{cls.__name__}"
    else:
        name = source[len('builtin:'):] if source.startswith('builtin:') else source
        cls = get_evaluation_class(name)
        source = f"builtin:{name}"

    try:
        evaluation = cls(**dict(params or {}))
    except TypeError as e:
        raise PolicyError(colorstr("red", f"Invalid parameters for evaluation `{source}`: {e}")) from e
    
    evaluation.id = str(id) if id else cls.name
    evaluation.source = source
    evaluation.source_sha256 = digest
    
    return evaluation



def default_policies() -> PolicyStack:
    """
    Policies of a run with no policy config: a `result_gain` policy in ``'patient-side'`` mode, so the staff never
    negotiates, and no `restrict` policy, so nothing is restricted.

    Returns:
        PolicyStack: The default policies.
    """
    return PolicyStack([build_policy('builtin:result_gain', {'mode': 'patient-side'})])



def _build_entries(entries: list[dict], kind: str, builder: Callable, base_dir: str, where: str) -> list:
    """
    Build the policy, metric or evaluation entries of one config file, in file order.

    Args:
        entries (list[dict]): The YAML list under `policies`, `metrics` or `evaluations` (empty when the key is absent).
        kind (str): ``'policies'``, ``'metrics'`` or ``'evaluations'``, for error messages.
        builder (Callable): `build_policy`, `build_metric` or `build_evaluation`.
        base_dir (str): Directory relative ``file:`` sources are resolved against (the YAML file's own directory).
        where (str): Label of the config, prefixed to error messages.

    Raises:
        PolicyError: If `entries` is not a list, or an entry is not a mapping, has unknown keys, has no `source` or
                     its `params` is not a mapping.

    Returns:
        list: The built objects.
    """
    # Example sources shown when an entry has none (only policies have built-ins)
    _SOURCE_EXAMPLES = {
        'policies': "`builtin:result_gain` or `file:<path.py>:<Class>`",
        'metrics': "`file:<path.py>:<Class>`, or `builtin:<name>` for a metric registered with `register_metric`",
        'evaluations': "`file:<path.py>:<Class>`, or `builtin:<name>` for an evaluation registered with `register_evaluation`",
    }

    if not isinstance(entries, list):
        raise PolicyError(colorstr("red", f"{where}: `{kind}` must be a list."))
    
    built = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PolicyError(colorstr("red", f"{where}: {kind}[{i}] must be a mapping, got {type(entry).__name__}."))
        
        unknown = set(entry) - POLICY_ENTRY_KEYS
        if unknown:
            raise PolicyError(colorstr("red", f"{where}: {kind}[{i}] has unknown keys {sorted(unknown)}; allowed: {sorted(POLICY_ENTRY_KEYS)}."))
        
        if 'source' not in entry:
            raise PolicyError(colorstr("red", f"{where}: {kind}[{i}] needs a `source` (e.g. {_SOURCE_EXAMPLES[kind]})."))
        
        params = entry.get('params') or {}
        if not isinstance(params, dict):
            raise PolicyError(colorstr("red", f"{where}: {kind}[{i}].params must be a mapping."))
        
        built.append(builder(entry['source'], params, entry.get('id'), base_dir))
    
    return built



def _entries_to_policies(data: dict, base_dir: str, where: str) -> tuple:
    """
    Build the policies, metrics and evaluations listed in one parsed policy config file, in file order.

    Args:
        data (dict): The parsed YAML of one policy config file (``{'api_version': 1, 'policies': [...], 'metrics': [...]}``).
        base_dir (str): Directory relative ``file:`` sources are resolved against (the YAML file's own directory).
        where (str): Label of the config (e.g. its path), prefixed to error messages.

    Raises:
        PolicyError: If the config is not a mapping, has unknown keys or a wrong `api_version`.

    Returns:
        tuple: ``(policies, metrics, evaluations)``, each in file order.
    """
    if not isinstance(data, dict):
        raise PolicyError(colorstr("red", f"{where}: a policy config must be a mapping, got {type(data).__name__}."))
    
    unknown = set(data) - POLICY_CONFIG_KEYS
    if unknown:
        raise PolicyError(colorstr("red", f"{where}: unknown policy config keys {sorted(unknown)}; allowed: {sorted(POLICY_CONFIG_KEYS)}."))
    
    if data.get('api_version') != POLICY_CONFIG_API_VERSION:
        raise PolicyError(colorstr("red", f"{where}: `api_version` must be {POLICY_CONFIG_API_VERSION}, got {data.get('api_version')!r}."))

    policies = _build_entries(data.get('policies') or [], 'policies', build_policy, base_dir, where)
    metrics = _build_entries(data.get('metrics') or [], 'metrics', build_metric, base_dir, where)
    evaluations = _build_entries(data.get('evaluations') or [], 'evaluations', build_evaluation, base_dir, where)
    
    return policies, metrics, evaluations



def load_policy_config(paths: Optional[Union[str, os.PathLike, list]] = None) -> PolicyStack:
    """
    Load one or more policy config YAML files into the run's policies.

    Several files are applied in the given order (priority order: the first file's policies come first), so
    independent policies such as `capacity_protection.yaml` and `tau.yaml` can be kept in separate files. Relative
    ``file:`` sources resolve against the directory of the YAML file that names them.

    Format of one file::

        api_version: 1
        policies:
          - id: tau                          # optional, defaults to the policy name
            source: builtin:result_gain      # or file:my_policies/urgent_keep.py:UrgentKeepPolicy
            params: {mode: threshold, negotiation_trigger_threshold: 1.0}
        metrics:                             # optional user-defined negotiation metrics (see `Metric`)
          - id: added_visit_days
            source: file:my_metrics/visits.py:AddedVisitDays
        evaluations:                         # optional user-defined evaluations of finished runs (see `Evaluation`)
          - id: negotiated_share
            source: file:my_evaluations/share.py:NegotiatedShare

    Args:
        paths (Optional[Union[str, os.PathLike, list]], optional): A policy config file, or a list of them in priority
            order. None (or an empty list) builds `default_policies()`. Defaults to None.

    Raises:
        PolicyError: If a file is missing or is not valid YAML.

    Returns:
        PolicyStack: The run's policies and metrics.
    """
    paths = _config_paths(paths)
    if not paths:
        return default_policies()

    policies, metrics, evaluations = [], [], []
    for path in paths:
        if not os.path.isfile(path):
            raise PolicyError(colorstr("red", f"Policy config not found: {path}"))
        
        with open(path) as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise PolicyError(colorstr("red", f"Invalid YAML in policy config {path}: {e}")) from e
        
        file_policies, file_metrics, file_evaluations = _entries_to_policies(data, os.path.dirname(path), path)
        policies.extend(file_policies)
        metrics.extend(file_metrics)
        evaluations.extend(file_evaluations)
    
    return PolicyStack(policies, metrics=metrics, evaluations=evaluations)



def snapshot_policy_config(paths: Union[str, os.PathLike, list],
                           output_dir: Union[str, os.PathLike]) -> Optional[list]:
    """
    Validate the policy config files before a run starts (so a broken config fails before any worker spawns) and copy
    them into `output_dir/policy/`, so the run folder alone holds the exact policies it used and can be re-run.

    Each config file becomes ``NN_<name>.yaml`` (``NN`` = its priority position). The Python file of every ``file:``
    source is copied next to it as ``NN_<file>.py`` and the copied YAML's ``source`` is rewritten to point at that copy
    (such a YAML is re-written from its parsed content, so its comments are not kept; a YAML with only ``builtin:``
    sources is copied byte for byte). Config files that already live in `output_dir/policy/` (e.g. when re-running
    from a saved ``args.yaml``) are used as they are. The snapshot is loaded once more at the end, to make sure it runs
    on its own.

    Args:
        paths (Union[str, os.PathLike, list]): A policy config file, or a list of them in priority order (the run's
                                               `schedule_task.policy_config`). An empty list does nothing.
        output_dir (Union[str, os.PathLike]): The run's output directory; the copies go to its `policy/` subdirectory.

    Returns:
        Optional[list]: Paths of the snapshot config files in priority order, as `output_dir/policy/<name>` (relative
                        when `output_dir` is), or None for an empty list.
    """
    paths = _config_paths(paths)
    if not paths:
        return None
    load_policy_config(paths)   # Validation only: fail on a broken original before anything is copied

    snapshot_dir = os.path.join(os.fspath(output_dir), 'policy')
    abs_snapshot_dir = os.path.abspath(snapshot_dir)
    os.makedirs(snapshot_dir, exist_ok=True)

    taken = set()
    copied_code = {}    # absolute path of a policy .py file -> its file name in the snapshot directory
    snapshot_paths = []
    for i, path in enumerate(paths):
        # Already a snapshot of this run folder: keep it as is
        if os.path.dirname(path) == abs_snapshot_dir:
            taken.add(os.path.basename(path))
            snapshot_paths.append(os.path.join(snapshot_dir, os.path.basename(path)))
            continue

        name = _unique_name(f"{i:02d}_{os.path.basename(path)}", taken)
        with open(path) as f:
            data = yaml.safe_load(f)

        # Copy the code of every `file:` source (policies, metrics and evaluations) and point the copied YAML at it
        rewritten = False
        for entry in [e for key in ('policies', 'metrics', 'evaluations') for e in (data.get(key) or [])]:
            source = entry.get('source')
            if not (isinstance(source, str) and source.startswith('file:')):
                continue
            code_path, class_name = _resolve_file_source(source[len('file:'):], os.path.dirname(path))
            if code_path not in copied_code:
                copied_code[code_path] = _unique_name(f"{i:02d}_{os.path.basename(code_path)}", taken)
                shutil.copyfile(code_path, os.path.join(snapshot_dir, copied_code[code_path]))
            entry['source'] = f"file:{copied_code[code_path]}:{class_name}"
            rewritten = True

        target = os.path.join(snapshot_dir, name)
        if rewritten:
            with open(target, 'w') as f:
                f.write(f"# Snapshot of {path}\n# `file:` sources point to the policy code copied next to this file.\n")
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        else:
            shutil.copyfile(path, target)
        snapshot_paths.append(target)

    load_policy_config(snapshot_paths)  # Validation only: the snapshot must load on its own
    return snapshot_paths



def pin_policy_config(schedule_task: Any, output_dir: Union[str, os.PathLike]) -> Optional[list]:
    """
    Snapshot a run's policy config into its output directory (see `snapshot_policy_config`) and point the config at
    the snapshot, so the saved ``args.yaml`` re-runs exactly these policies and the run's workers load them too.

    ``schedule_task.policy_config`` is replaced by the snapshot paths, and the paths it had are kept under
    ``schedule_task.policy_config_original`` (left untouched if already set, e.g. when re-running a saved
    ``args.yaml``). Call it before saving ``args.yaml``.

    Args:
        schedule_task (Any): The ``schedule_task`` section of the simulation config (a mutable mapping, e.g. an
                             `sconf` config node).
        output_dir (Union[str, os.PathLike]): The run's output directory.

    Returns:
        Optional[list]: The snapshot paths now in ``schedule_task.policy_config``, or None when no policy config is given
                        (the config is then left unchanged).
    """
    original = schedule_task.get('policy_config')
    if not original:    # no policy config: the run uses `default_policies()`, nothing to snapshot
        return None
    snapshot = snapshot_policy_config(original, output_dir)
    if snapshot is None:
        return None
    if schedule_task.get('policy_config_original') is None:
        originals = [original] if isinstance(original, (str, os.PathLike)) else list(original)
        schedule_task.policy_config_original = [os.fspath(p) for p in originals]
    schedule_task.policy_config = snapshot
    return snapshot
