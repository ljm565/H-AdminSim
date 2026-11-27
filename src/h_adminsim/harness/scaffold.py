import re
import yaml
from string import Template
from importlib import resources
from typing import Iterable, Optional

from h_adminsim.registry.harness import (
    HARNESS_TEMPLATE_DIR,
    HARNESS_POLICY_TEMPLATE,
    HARNESS_HOOK_TEMPLATES,
    HARNESS_METRIC_TEMPLATE,
)
from h_adminsim.registry.policy import (
    POLICY_HOOKS,
    POLICY_FILE_PREFIX,
    POLICY_CONFIG_API_VERSION,
    POLICY_REQUIRES_PATIENT_PREFIX,
    POLICY_REQUIRES_METRIC_PREFIX,
)

# Names each hook's code imports: (from h_adminsim.policy, from typing)
_HOOK_IMPORTS = {
    'restrict': (('RestrictContext',), ()),
    'review': (('Decision', 'ReviewContext'), ('Optional',)),
}

# Python type names of the scalar parameter types a template accepts
_PARAM_TYPES = {bool: 'bool', int: 'int', float: 'float', str: 'str'}



def _template(file_name: str) -> str:
    """
    Text of a scaffold template under ``h_adminsim/assets/harness/templates``, without its final newline.

    Args:
        file_name (str): The template file name.

    Returns:
        str: The template text.
    """
    path = resources.files('h_adminsim').joinpath('assets', 'harness', HARNESS_TEMPLATE_DIR, file_name)
    return path.read_text().rstrip('\n')



def _snake_case(class_name: str) -> str:
    """
    Policy name from a class name: ``UrgentKeepPolicy`` -> ``urgent_keep``.

    Args:
        class_name (str): The class name.

    Returns:
        str: The snake-case name, without a trailing ``_policy``.
    """
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
    return name[:-len('_policy')] if name.endswith('_policy') and name != '_policy' else name



def _check_names(class_name: str, name: str, requires: list, params: dict) -> None:
    """
    Check the names, `requires` entries and parameters of a policy or metric template.

    Args:
        class_name (str): The class name.
        name (str): The policy or metric name.
        requires (list): The `requires` entries.
        params (dict): Parameters with their default values.

    Raises:
        ValueError: If `class_name` or `name` is not an identifier, a `requires` entry is malformed, or a parameter is
                    not a scalar.
    """
    if not class_name.isidentifier() or not name.isidentifier():
        raise ValueError(f"`class_name` and `name` must be Python identifiers, got {class_name!r}, {name!r}")
    prefixes = (POLICY_REQUIRES_PATIENT_PREFIX, POLICY_REQUIRES_METRIC_PREFIX)
    bad = [r for r in requires if not any(r.startswith(p) and len(r) > len(p) for p in prefixes)]
    if bad:
        raise ValueError(f"`requires` entries must look like 'patient.<field>' or 'metric.<id>', got {bad}")
    bad = [k for k, v in params.items() if type(v) not in _PARAM_TYPES or not k.isidentifier()]
    if bad:
        raise ValueError(f"Parameters must be identifiers with bool / int / float / str defaults; edit the generated "
                         f"code for structured ones: {bad}")



def _class_parts(requires: list, params: dict) -> dict:
    """
    Template values shared by policy and metric classes: the `Fields:` docstring block, the dataclass fields and the
    `requires` tuple.

    Args:
        requires (list): The `requires` entries.
        params (dict): Parameters with their default values.

    Returns:
        dict: ``{'fields_doc', 'fields', 'requires'}``.
    """
    fields_doc = ''.join(f"\n        {k}: TODO." for k in params)
    fields_code = ''.join(f"\n    {k}: {_PARAM_TYPES[type(v)]} = {v!r}" for k, v in params.items())
    return {'fields_doc': f"\n\n    Fields:{fields_doc}" if params else '',
            'fields': f"\n{fields_code}" if params else '',
            'requires': f"({', '.join(repr(r) for r in requires)}{',' if len(requires) == 1 else ''})"}



def _config(kind: str, name: str, file_name: str, class_name: str, params: dict) -> str:
    """
    A policy config YAML with one entry.

    Args:
        kind (str): ``'policies'`` or ``'metrics'``.
        name (str): The entry id.
        file_name (str): The Python file of the class.
        class_name (str): The class name.
        params (dict): The entry's parameters (left out when empty).

    Returns:
        str: The YAML text.
    """
    entry = {'id': name, 'source': f"{POLICY_FILE_PREFIX}{file_name}:{class_name}"}
    if params:
        entry['params'] = params
    return yaml.safe_dump({'api_version': POLICY_CONFIG_API_VERSION, kind: [entry]}, sort_keys=False)



def policy_template(class_name: str,
                    hooks: Iterable[str] = ('review',),
                    name: Optional[str] = None,
                    requires: Iterable[str] = (),
                    params: Optional[dict] = None,
                    file_name: Optional[str] = None) -> dict:
    """
    A runnable policy skeleton and the policy config that loads it. The skeleton passes `validate_policy` as is (its
    hooks block nothing and abstain), so the TODOs can be filled in step by step.

    Args:
        class_name (str): The policy class name (e.g. ``'UrgentKeepPolicy'``).
        hooks (Iterable[str], optional): Hooks to implement. Defaults to ('review',).
        name (Optional[str], optional): The policy `name` (and config id). Defaults to the snake-case class name.
        requires (Iterable[str], optional): The `requires` entries (``'patient.<field>'`` / ``'metric.<id>'``).
                                            Defaults to ().
        params (Optional[dict], optional): Parameters with their default values (bool, int, float or str), declared as
                                           dataclass fields and listed in the config. Defaults to None.
        file_name (Optional[str], optional): The Python file name used in the config's source. Defaults to
                                             ``<name>.py``.

    Raises:
        ValueError: If a hook is unknown or none is given.

    Returns:
        dict: ``{'file_name', 'code', 'config'}`` (the Python source and the config YAML text).
    """
    requested = set(hooks)
    hooks = [h for h in POLICY_HOOKS if h in requested]
    if not hooks or requested - set(POLICY_HOOKS):
        raise ValueError(f"`hooks` must be a non-empty subset of {list(POLICY_HOOKS)}")
    name = name or _snake_case(class_name)
    requires, params = list(requires), dict(params or {})
    _check_names(class_name, name, requires, params)
    file_name = file_name or f"{name}.py"

    api_imports = ['Policy'] + sorted({n for h in hooks for n in _HOOK_IMPORTS[h][0]})
    typing_imports = ['ClassVar'] + sorted({n for h in hooks for n in _HOOK_IMPORTS[h][1]})
    methods = '\n'.join(f"\n{_template(HARNESS_HOOK_TEMPLATES[h])}\n" for h in hooks)
    code = Template(_template(HARNESS_POLICY_TEMPLATE)).substitute(
        typing_imports=', '.join(typing_imports), api_imports=', '.join(api_imports), class_name=class_name,
        name=repr(name), methods=methods, **_class_parts(requires, params),
    )
    return {'file_name': file_name, 'code': code, 'config': _config('policies', name, file_name, class_name, params)}



def metric_template(class_name: str,
                    name: Optional[str] = None,
                    requires: Iterable[str] = (),
                    params: Optional[dict] = None,
                    file_name: Optional[str] = None) -> dict:
    """
    A user-defined metric skeleton (`Metric`, computed from a `ReviewContext`) and the policy config that lists it under
    ``metrics:``. Policies read its value as ``ctx.metrics[<name>]`` after declaring ``'metric.<name>'``.

    Args:
        class_name (str): The metric class name (e.g. ``'TestDaySpread'``).
        name (Optional[str], optional): The metric id (its key in ``ctx.metrics``). Defaults to the snake-case class
                                        name.
        requires (Iterable[str], optional): The `requires` entries (``'patient.<field>'`` / ``'metric.<id>'``).
                                            Defaults to ().
        params (Optional[dict], optional): Parameters with their default values (bool, int, float or str). Defaults to
                                           None.
        file_name (Optional[str], optional): The Python file name used in the config's source. Defaults to
                                             ``<name>_metric.py`` (a file of its own, next to the policy reading it).

    Returns:
        dict: ``{'file_name', 'code', 'config'}`` (the Python source and the config YAML text).
    """
    name = name or _snake_case(class_name)
    requires, params = list(requires), dict(params or {})
    _check_names(class_name, name, requires, params)
    file_name = file_name or f"{name}_metric.py"
    code = Template(_template(HARNESS_METRIC_TEMPLATE)).substitute(
        typing_imports='ClassVar, Optional', class_name=class_name, name=repr(name), **_class_parts(requires, params),
    )
    return {'file_name': file_name, 'code': code, 'config': _config('metrics', name, file_name, class_name, params)}
