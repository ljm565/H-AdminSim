import os
import ast
import inspect
import importlib
from typing import Union

import yaml

from h_adminsim.policy.builder import _config_paths, _resolve_file_source
from h_adminsim.registry.errors import PolicyError
from h_adminsim.registry.policy import POLICY_FILE_PREFIX
from h_adminsim.registry.harness import (
    HARNESS_CODE_FORBIDDEN_ATTRS,
    HARNESS_CODE_FORBIDDEN_NAMES,
    HARNESS_CODE_H_ADMINSIM_NAMES,
    HARNESS_CODE_MODULES,
)

_REASON = "policy code may not reach the environment, files, the network or other processes"
_NAME_LOOKUPS = ('getattr', 'hasattr')



def _allowed_module(name: str) -> bool:
    """
    Whether policy code may import from a module: one of `HARNESS_CODE_MODULES` or a submodule of one.

    Args:
        name (str): Dotted module name.

    Returns:
        bool: Whether it is allowed.
    """
    return any(name == m or name.startswith(f'{m}.') for m in HARNESS_CODE_MODULES)


def _import_issues(node: Union[ast.Import, ast.ImportFrom]) -> list[str]:
    """
    What is not allowed in one import statement: a module object (`import m`), a module outside
    `HARNESS_CODE_MODULES`, a relative or `*` import, or an H-AdminSim name outside `HARNESS_CODE_H_ADMINSIM_NAMES`.

    Args:
        node (Union[ast.Import, ast.ImportFrom]): The statement.

    Returns:
        list[str]: One problem per item (empty: allowed).
    """
    if isinstance(node, ast.Import):
        return [f"`import {a.name}`: import the names you use instead (`from {a.name} import ...`)"
                if _allowed_module(a.name) else f"`import {a.name}` is not allowed" for a in node.names]
    if node.level:
        return ["relative imports are not allowed (a policy file is loaded on its own)"]
    module = node.module or ''
    if not _allowed_module(module):
        return [f"`from {module} import ...` is not allowed"]
    try:
        loaded = importlib.import_module(module)
    except ImportError:
        return [f"`from {module} import ...`: no such module"]
    issues = []
    for alias in node.names:
        if alias.name == '*':
            issues.append(f"`from {module} import *` is not allowed; name what you use")
        elif inspect.ismodule(getattr(loaded, alias.name, None)):
            issues.append(f"`from {module} import {alias.name}` imports a module; import the names you use from it")
        elif module.split('.')[0] == 'h_adminsim' and not alias.name.isupper() \
                and alias.name not in HARNESS_CODE_H_ADMINSIM_NAMES:
            issues.append(f"`{alias.name}` is not part of the policy API policy code may import")
    return issues


def check_code(source: str, filename: str = '<policy>') -> list[tuple[int, str]]:
    """
    Check user policy code (a `file:` source) before it is imported: names may be imported only from
    `HARNESS_CODE_MODULES` (never a module object; from H-AdminSim, `HARNESS_CODE_H_ADMINSIM_NAMES` and constants),
    and `HARNESS_CODE_FORBIDDEN_NAMES`, `HARNESS_CODE_FORBIDDEN_ATTRS` and `getattr` / `hasattr` with a dunder name, or
    with a computed name on anything but `self`, are not allowed. Simulations run policy code next to the
    agents' API keys; this keeps it from reaching them (a best-effort guard, not a sandbox).

    Args:
        source (str): The file's text.
        filename (str, optional): Its name, for syntax errors. Defaults to '<policy>'.

    Returns:
        list[tuple[int, str]]: ``(line, problem)`` per finding (empty: allowed). A syntax error is reported as one.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        return [(e.lineno or 0, f"syntax error: {e.msg}")]

    found, checked_lookups = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _NAME_LOOKUPS:
            checked_lookups.add(id(node.func))
            target = node.args[0] if node.args else None
            name = node.args[1] if len(node.args) > 1 else None
            literal = isinstance(name, ast.Constant) and isinstance(name.value, str)
            on_self = isinstance(target, ast.Name) and target.id == 'self'
            if (literal and (name.value.startswith('__') or name.value in HARNESS_CODE_FORBIDDEN_ATTRS)) \
                    or (not literal and not on_self):
                found.append((node.lineno, f"`{node.func.id}` only with a plain attribute name written out "
                              f"(`getattr(obj, 'name', None)`), or with a computed name on `self`; for a mapping use "
                              f"`obj.get(key)`"))
    for node in ast.walk(tree):
        line = getattr(node, 'lineno', 0)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.extend((line, issue) for issue in _import_issues(node))
        elif isinstance(node, ast.Name) and node.id in HARNESS_CODE_FORBIDDEN_NAMES:
            found.append((line, f"`{node.id}` is not allowed"))
        elif isinstance(node, ast.Name) and node.id in _NAME_LOOKUPS and id(node) not in checked_lookups:
            found.append((line, f"`{node.id}` may only be called directly, with a plain attribute name"))
        elif isinstance(node, ast.Attribute) and node.attr in HARNESS_CODE_FORBIDDEN_ATTRS:
            hint = "; use an f-string" if node.attr in ('format', 'format_map') else ''
            found.append((line, f"the attribute `.{node.attr}` is not allowed{hint}"))
    return sorted(set(found))


def config_code_files(paths: Union[str, os.PathLike, list]) -> list[str]:
    """
    The user code files policy configs load (`file:` sources of their policies, metrics and evaluations), read from
    the YAML without importing anything.

    Args:
        paths (Union[str, os.PathLike, list]): A policy config, or a list of them (``builtin:<name>`` entries too).

    Raises:
        PolicyError: If a config is missing or not valid YAML, or a `file:` source is malformed or missing.

    Returns:
        list[str]: Absolute paths, each once, in the order found.
    """
    files = []
    for path in _config_paths(paths):
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except OSError as e:
            raise PolicyError(f"Policy config not readable: {path} ({e})") from e
        except yaml.YAMLError as e:
            raise PolicyError(f"Invalid YAML in policy config {path}: {e}") from e
        if not isinstance(data, dict):
            continue    # not a config; loading it reports the problem
        for key, kind in (('policies', 'policy'), ('metrics', 'metric'), ('evaluations', 'evaluation')):
            for entry in data.get(key) or []:
                source = entry.get('source') if isinstance(entry, dict) else None
                if isinstance(source, str) and source.startswith(POLICY_FILE_PREFIX):
                    file, _ = _resolve_file_source(source[len(POLICY_FILE_PREFIX):], os.path.dirname(path), kind)
                    if file not in files:
                        files.append(file)
    return files


def config_code_issues(paths: Union[str, os.PathLike, list]) -> list[dict]:
    """
    `check_code` for every user code file policy configs load, without importing any of them.

    Args:
        paths (Union[str, os.PathLike, list]): A policy config, or a list of them.

    Raises:
        PolicyError: As `config_code_files`.

    Returns:
        list[dict]: ``{'file', 'line', 'message'}`` per finding (empty: every file is allowed).
    """
    issues = []
    for file in config_code_files(paths):
        with open(file) as f:
            source = f.read()
        issues.extend({'file': file, 'line': line, 'message': f"{message} ({_REASON})"}
                      for line, message in check_code(source, file))
    return issues
