import sys
import json
import inspect
from importlib import resources
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Optional, Union

from h_adminsim.registry.harness import HARNESS_SYSTEM_PROMPT
from h_adminsim.registry.policy import (
    POLICY_HOOKS,
    POLICY_ACTIONS,
    POLICY_SUPPORTED_OBJECTIVES,
    POLICY_RESERVED_RECORD_KEYS,
    POLICY_CONFIG_API_VERSION,
    POLICY_REQUIRES_PATIENT_PREFIX,
    POLICY_REQUIRES_METRIC_PREFIX,
    POLICY_BUILTIN_METRIC_IDS,
    POLICY_ASSET_PREFIX,
)
from h_adminsim.policy.base import Policy, Decision
from h_adminsim.policy.context import RestrictContext, ReviewContext, PatientView, MetricView
from h_adminsim.policy.stack import PolicyStack
from h_adminsim.policy.metric import Metric
from h_adminsim.policy.evaluation import Evaluation, RunData
from h_adminsim.policy.builder import builtin_policy_config_dir, load_policy_config
from h_adminsim.policy.registry import registered_policies
from h_adminsim.tools.negotiation_metrics import NegotiationMetrics
from h_adminsim.harness.validate import load_patient_records
from h_adminsim.harness.prompts import prompt_contract
from h_adminsim.utils.policy_utils import available_patient_fields

_IMPORT_LINE = "from h_adminsim.policy import Policy, Decision, PolicyError, RestrictContext, ReviewContext"

# Built-in metric id -> `NegotiationMetrics` property documenting it
_METRIC_PROPERTIES = {'gpc': 'gpc', 'tcl': 'tcl', 'G': 'G', 'R': 'R', 'U': 'U', 'U_pref': 'u_pref', 'U_thr': 'u_thr'}



def system_prompt() -> str:
    """
    System prompt for an LLM writing policies with the harness tools (the workflow, and that conflicts between policies
    are left to the user).

    Returns:
        str: The prompt text.
    """
    return resources.files('h_adminsim.assets.harness').joinpath(HARNESS_SYSTEM_PROMPT).read_text()



def bundled_policy_dir() -> str:
    """
    Directory of the policy configs shipped with the package (``tau.yaml``, ``capacity_protection.yaml``), which a new
    policy can be listed with under ``schedule_task.policy_config`` (as ``builtin:<name>`` or by path).

    Returns:
        str: The directory path.
    """
    return builtin_policy_config_dir()



def _method(cls: type, name: str) -> dict:
    """
    Signature and docstring of a method.

    Args:
        cls (type): The class.
        name (str): The method name.

    Returns:
        dict: ``{'signature', 'doc'}``.
    """
    member = getattr(cls, name)
    return {'signature': f"{name}{inspect.signature(member)}", 'doc': inspect.getdoc(member) or ''}



def _config_format() -> str:
    """
    The policy config file format, taken from `load_policy_config`'s docstring.

    Returns:
        str: The YAML example.
    """
    doc = inspect.getdoc(load_policy_config) or ''
    block = doc.split('Format of one file::', 1)[-1].split('\nArgs:', 1)[0]
    return inspect.cleandoc(block)



def _default_params(cls: type) -> dict:
    """
    Default parameters of a dataclass policy.

    Args:
        cls (type): The policy class.

    Returns:
        dict: ``{field: default}`` (None for a field without default).
    """
    if not is_dataclass(cls):
        return {}
    return {f.name: (None if f.default is MISSING else f.default) for f in fields(cls)}



def _module_source(cls: type) -> Optional[str]:
    """
    Source code of the module defining a class.

    Args:
        cls (type): The class.

    Returns:
        Optional[str]: The source, or None if unavailable.
    """
    try:
        return inspect.getsource(sys.modules[cls.__module__])
    except (KeyError, OSError, TypeError):
        return None



def _rules() -> list[str]:
    """
    The authoring rules a policy must follow.

    Returns:
        list[str]: One rule per item.
    """
    return [
        f"Import the policy API from `h_adminsim.policy` (`{_IMPORT_LINE}`).",
        f"Subclass `Policy`, decorate it with `@dataclass`, set `name: ClassVar[str]` and implement at least one hook "
        f"({', '.join(f'`{h}`' for h in POLICY_HOOKS)}).",
        "Configuration lives in the dataclass fields; the config entry's `params` sets them. Check them in `__post_init__` "
        "and raise `PolicyError` on bad values.",
        "Hooks must be pure functions of `ctx`: no state on `self` or in globals, no randomness, no I/O.",
        f"Declare every patient field a hook reads as `'{POLICY_REQUIRES_PATIENT_PREFIX}<field>'` and every metric as "
        f"`'{POLICY_REQUIRES_METRIC_PREFIX}<id>'` in `requires: ClassVar[tuple]` (`preference` needs none). Reading an "
        f"undeclared key raises.",
        "`restrict` returns `(test_code, date)` pairs (date `'YYYY-MM-DD'`) taken from `ctx.test_capacity` keys.",
        f"`review` returns a `Decision` or None to abstain. Non-`keep` decisions must use an objective in "
        f"{list(POLICY_SUPPORTED_OBJECTIVES)}; `negotiate` needs an existing prompt file.",
        f"A decision's `record` keys may not reuse {sorted(POLICY_RESERVED_RECORD_KEYS)} or user metric ids.",
        f"Files a policy reads (e.g. its own negotiation prompt) are declared under `assets:` in its config and passed "
        f"as parameter values `'{POLICY_ASSET_PREFIX}<id>'` (resolved to the file path; copied into each run).",
    ]



def policy_api_spec(agent_data_path: Optional[str] = None, fmt: str = 'markdown', include_sources: bool = True) -> Union[str, dict]:
    """
    Description of the policy API for an LLM writing policies, built from the installed package (docstrings,
    signatures, constants and the built-in policies' source), so it always matches the installed version.

    Args:
        agent_data_path (Optional[str], optional): Agent data whose patient fields (and one example record) are listed;
                                                   None leaves them out. Defaults to None.
        fmt (str, optional): ``'markdown'`` or ``'dict'``. Defaults to 'markdown'.
        include_sources (bool, optional): Whether to include the built-in policies' source code as examples (about
                                          half the text). Defaults to True.

    Raises:
        ValueError: If `fmt` is unknown.

    Returns:
        Union[str, dict]: The description.
    """
    if fmt not in ('markdown', 'dict'):
        raise ValueError(f"`fmt` must be 'markdown' or 'dict', got {fmt!r}")

    if agent_data_path:
        records, origin = load_patient_records(agent_data_path), f"agent data {agent_data_path}"
    else:
        records, origin = [], 'no agent data given (see `describe_data`)'

    spec = {
        'api_version': POLICY_CONFIG_API_VERSION,
        'import': _IMPORT_LINE,
        'rules': _rules(),
        'policy': inspect.getdoc(Policy),
        'hooks': {
            'restrict': {**_method(Policy, 'restrict'), 'context': {
                'name': 'RestrictContext', 'doc': inspect.getdoc(RestrictContext),
                'methods': [_method(RestrictContext, 'lead_hours')]}},
            'review': {**_method(Policy, 'review'), 'context': {
                'name': 'ReviewContext', 'doc': inspect.getdoc(ReviewContext), 'methods': []}},
        },
        'decision': {
            'doc': inspect.getdoc(Decision),
            'constructors': [_method(Decision, n) for n in POLICY_ACTIONS],
            'actions': list(POLICY_ACTIONS),
            'objectives': list(POLICY_SUPPORTED_OBJECTIVES),
            'reserved_record_keys': sorted(POLICY_RESERVED_RECORD_KEYS),
        },
        'metrics_params': _method(Policy, 'metrics_params'),
        'composition': inspect.getdoc(PolicyStack),
        'requires': {
            'patient_prefix': POLICY_REQUIRES_PATIENT_PREFIX,
            'metric_prefix': POLICY_REQUIRES_METRIC_PREFIX,
            'builtin_metrics': sorted(POLICY_BUILTIN_METRIC_IDS),
            'metrics_overview': inspect.getdoc(NegotiationMetrics),
            'builtin_metric_docs': {m: inspect.getdoc(getattr(NegotiationMetrics, _METRIC_PROPERTIES[m])) or ''
                                    for m in sorted(POLICY_BUILTIN_METRIC_IDS)},
            'patient_fields': sorted(available_patient_fields(records)),
            'patient_fields_origin': origin,
            'patient_example': records[0] if records else None,
            'patient_view': inspect.getdoc(PatientView),
            'metric_view': inspect.getdoc(MetricView),
        },
        'metric': {'doc': inspect.getdoc(Metric), 'compute': _method(Metric, 'compute')},
        'evaluation': {'doc': inspect.getdoc(Evaluation), 'evaluate': _method(Evaluation, 'evaluate'),
                       'run_data': inspect.getdoc(RunData), 'bookings': inspect.getdoc(RunData.bookings)},
        'config_format': _config_format(),
        'negotiation_prompts': prompt_contract(),
        'builtin_policies': [
            {'name': name, 'hooks': list(cls.hooks()), 'requires': list(cls.requires),
             'params': _default_params(cls), 'source': _module_source(cls) if include_sources else None}
            for name, cls in registered_policies().items()
        ],
    }
    return spec if fmt == 'dict' else _to_markdown(spec)



def _code(text: str, lang: str = 'python') -> list[str]:
    """
    A fenced code block.

    Args:
        text (str): The code.
        lang (str, optional): The block language. Defaults to 'python'.

    Returns:
        list[str]: The block's lines.
    """
    return [f"```{lang}", text.rstrip(), "```"]



def _json(value: Any) -> str:
    """
    Indented JSON of a value (non-JSON values as strings).

    Args:
        value (Any): The value.

    Returns:
        str: The JSON text.
    """
    return json.dumps(value, indent=1, ensure_ascii=False, default=str)



def _to_markdown(spec: dict) -> str:
    """
    Render `policy_api_spec`'s dict as Markdown.

    Args:
        spec (dict): The spec dict.

    Returns:
        str: The Markdown text.
    """
    out = [f"# H-AdminSim policy API (api_version {spec['api_version']})", "", "## Rules"]
    out += [f"- {r}" for r in spec['rules']] + ["", "## Policy base class", spec['policy'], ""]

    for hook, h in spec['hooks'].items():
        ctx = h['context']
        out += [f"## Hook `{hook}`"] + _code(f"def {h['signature']}: ...") + [h['doc'], "", f"### `{ctx['name']}`", ctx['doc'], ""]
        for m in ctx['methods']:
            out += _code(f"def {m['signature']}: ...") + [m['doc'], ""]

    d = spec['decision']
    out += ["## `Decision`", d['doc'], ""]
    for m in d['constructors']:
        out += _code(f"Decision.{m['signature']}") + [m['doc'], ""]
    out += [f"Supported objectives: {d['objectives']}", f"Reserved record keys: {d['reserved_record_keys']}", ""]

    m = spec['metrics_params']
    out += ["## `metrics_params` (optional, `review` policies only)"] + _code(f"def {m['signature']}: ...") + [m['doc'], ""]
    out += ["## Composition (`PolicyStack`)", spec['composition'], ""]

    r = spec['requires']
    out += ["## `requires`",
            f"- Patient fields (`{r['patient_prefix']}<field>`), from {r['patient_fields_origin']}: {r['patient_fields']}",
            f"- Built-in metrics (`{r['metric_prefix']}<id>`): {r['builtin_metrics']}", "",
            "### Built-in metrics", r['metrics_overview'], ""]
    out += [f"- `{m}`: {' '.join(doc.split())}" for m, doc in r['builtin_metric_docs'].items()]
    out += ["",
            "### `ctx.patient`", r['patient_view'], "", "### `ctx.metrics`", r['metric_view'], ""]
    if r['patient_example'] is not None:
        out += ["Example patient record:"] + _code(_json(r['patient_example']), 'json') + [""]

    out += ["## User-defined metrics (`Metric`)", spec['metric']['doc'], ""]
    out += _code(f"def {spec['metric']['compute']['signature']}: ...") + [spec['metric']['compute']['doc'], ""]
    e = spec['evaluation']
    out += ["## User-defined evaluations (`Evaluation`)", e['doc'], ""] + _code(f"def {e['evaluate']['signature']}: ...")
    out += [e['evaluate']['doc'], "", "### `RunData`", e['run_data'], "", "`RunData.bookings`: " + e['bookings'], ""]
    out += ["## Policy config file"] + _code(spec['config_format'], 'yaml') + [""]
    out += ["## Negotiation prompts"] + [f"- {r}" for r in spec['negotiation_prompts']] + [""]

    out += ["## Built-in policies (examples)"]
    for p in spec['builtin_policies']:
        out += [f"### `{p['name']}` (source `builtin:{p['name']}`)",
                f"hooks: {p['hooks']}, requires: {p['requires']}, default params: {_json(p['params'])}"]
        if p['source']:
            out += _code(p['source'])
        out += [""]
    return '\n'.join(out)
