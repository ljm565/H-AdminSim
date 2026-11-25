from __future__ import annotations

from h_adminsim.utils import colorstr
from h_adminsim.registry.errors import PolicyError
from h_adminsim.registry.policy import POLICY_REQUIRES_PATIENT_PREFIX, POLICY_REQUIRES_METRIC_PREFIX
from h_adminsim.policy.base import Policy
from h_adminsim.policy.metric import Metric
from h_adminsim.policy.evaluation import Evaluation
from h_adminsim.policy.builtin import ResultGainPolicy, CapacityProtectionPolicy



def _validate_requires(cls, kind: str):
    """
    Check the format of a policy / metric class's `requires`.

    Args:
        cls: The policy or metric class.
        kind (str): ``'Policy'`` or ``'Metric'``, for the error message.

    Raises:
        PolicyError: If `requires` is not a tuple of ``patient.<field>`` / ``metric.<id>`` strings.
    """
    prefixes = (POLICY_REQUIRES_PATIENT_PREFIX, POLICY_REQUIRES_METRIC_PREFIX)
    requires = cls.requires
    if not isinstance(requires, (tuple, list)) or not all(
            isinstance(r, str) and any(r.startswith(p) and len(r) > len(p) for p in prefixes) for r in requires):
        raise PolicyError(colorstr("red", f"{kind} `{cls.name}`.requires must be a tuple of "
                                          f"'{POLICY_REQUIRES_PATIENT_PREFIX}<field>' / '{POLICY_REQUIRES_METRIC_PREFIX}<id>' "
                                          f"strings, got {requires!r}."))



# ----------------------------------------------------------------------------------------------------------------------
# Policies
# ----------------------------------------------------------------------------------------------------------------------
_REGISTRY: dict = {cls.name: cls for cls in (ResultGainPolicy, CapacityProtectionPolicy)}



def validate_policy_class(cls) -> type:
    """
    Check that `cls` can serve as a policy.

    Args:
        cls: The candidate class.

    Raises:
        PolicyError: If `cls` is not a `Policy` subclass, has no `name`, implements no hook, or declares metrics in
                     `requires` without implementing `review` (only `review` sees the metrics).

    Returns:
        type: `cls`.
    """
    if not (isinstance(cls, type) and issubclass(cls, Policy)):
        raise PolicyError(colorstr("red", f"`{cls}` is not a Policy subclass."))
    
    if not cls.name:
        raise PolicyError(colorstr("red", f"Policy class `{cls.__qualname__}` needs a non-empty `name`."))
    
    if not cls.hooks():
        raise PolicyError(colorstr("red", f"Policy `{cls.name}` implements no hook (`restrict` / `review`)."))
    
    _validate_requires(cls, 'Policy')
    if not cls.implements('review') and any(r.startswith(POLICY_REQUIRES_METRIC_PREFIX) for r in cls.requires):
        raise PolicyError(colorstr("red", f"Policy `{cls.name}` declares metrics in `requires` but does not implement "
                                          f"`review`, the only hook that sees them."))
    return cls



def register_policy(cls: type) -> type:
    """
    Register a policy class under its `name` so `builtin:<name>` sources can refer to it. Usable as a decorator.

    Args:
        cls (type): The policy class.

    Raises:
        PolicyError: If the name is taken by another class.

    Returns:
        type: `cls`.
    """
    validate_policy_class(cls)
    if _REGISTRY.get(cls.name, cls) is not cls:
        raise PolicyError(colorstr("red", f"Policy name `{cls.name}` is already registered by `{_REGISTRY[cls.name].__qualname__}`."))
    _REGISTRY[cls.name] = cls
    return cls



def get_policy_class(name: str) -> type:
    """
    The registered policy class named `name`.

    Args:
        name (str): The policy name.

    Raises:
        PolicyError: If no policy is registered under `name`.

    Returns:
        type: The policy class.
    """
    if name not in _REGISTRY:
        raise PolicyError(colorstr("red", f"Unknown policy `{name}`; choose one of {sorted(_REGISTRY)}."))
    return _REGISTRY[name]



def registered_policies() -> dict:
    """
    Every registered policy class (the built-ins and those added with `register_policy`).

    Returns:
        dict: ``{name: policy class}`` (a copy).
    """
    return dict(_REGISTRY)



# ----------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------
_METRIC_REGISTRY: dict = {}



def validate_metric_class(cls) -> type:
    """
    Check that `cls` can serve as a metric.

    Args:
        cls: The candidate class.

    Raises:
        PolicyError: If `cls` is not a `Metric` subclass, has no `name`, or does not implement `compute`.

    Returns:
        type: `cls`.
    """
    if not (isinstance(cls, type) and issubclass(cls, Metric)):
        raise PolicyError(colorstr("red", f"`{cls}` is not a Metric subclass."))
    
    if not cls.name:
        raise PolicyError(colorstr("red", f"Metric class `{cls.__qualname__}` needs a non-empty `name`."))
    
    if cls.compute is Metric.compute:
        raise PolicyError(colorstr("red", f"Metric `{cls.name}` does not implement `compute`."))
    
    _validate_requires(cls, 'Metric')
    return cls



def register_metric(cls: type) -> type:
    """
    Register a metric class under its `name` so `builtin:<name>` sources can refer to it. Usable as a decorator.

    Args:
        cls (type): The metric class.

    Raises:
        PolicyError: If the name is taken by another class.

    Returns:
        type: `cls`.
    """
    validate_metric_class(cls)
    if _METRIC_REGISTRY.get(cls.name, cls) is not cls:
        raise PolicyError(colorstr("red", f"Metric name `{cls.name}` is already registered by `{_METRIC_REGISTRY[cls.name].__qualname__}`."))
    _METRIC_REGISTRY[cls.name] = cls
    return cls



def get_metric_class(name: str) -> type:
    """
    The registered metric class named `name`.

    Args:
        name (str): The metric name.

    Raises:
        PolicyError: If no metric is registered under `name`.

    Returns:
        type: The metric class.
    """
    if name not in _METRIC_REGISTRY:
        raise PolicyError(colorstr("red", f"Unknown metric `{name}`; registered: {sorted(_METRIC_REGISTRY)}. "
                                          f"Use `file:<path.py>:<ClassName>` for a metric defined in a file."))
    return _METRIC_REGISTRY[name]



def registered_metrics() -> dict:
    """
    Every metric class added with `register_metric` (there are no built-in metrics).

    Returns:
        dict: ``{name: metric class}`` (a copy).
    """
    return dict(_METRIC_REGISTRY)



# ----------------------------------------------------------------------------------------------------------------------
# Evaluations
# ----------------------------------------------------------------------------------------------------------------------
_EVALUATION_REGISTRY: dict = {}



def validate_evaluation_class(cls) -> type:
    """
    Check that `cls` can serve as an evaluation.

    Args:
        cls: The candidate class.

    Raises:
        PolicyError: If `cls` is not an `Evaluation` subclass, has no `name`, or does not implement `evaluate`.

    Returns:
        type: `cls`.
    """
    if not (isinstance(cls, type) and issubclass(cls, Evaluation)):
        raise PolicyError(colorstr("red", f"`{cls}` is not an Evaluation subclass."))
    
    if not cls.name:
        raise PolicyError(colorstr("red", f"Evaluation class `{cls.__qualname__}` needs a non-empty `name`."))
    
    if cls.evaluate is Evaluation.evaluate:
        raise PolicyError(colorstr("red", f"Evaluation `{cls.name}` does not implement `evaluate`."))
    
    return cls




def register_evaluation(cls: type) -> type:
    """
    Register an evaluation class under its `name` so `builtin:<name>` sources can refer to it. Usable as a decorator.

    Args:
        cls (type): The evaluation class.

    Raises:
        PolicyError: If the name is taken by another class.

    Returns:
        type: `cls`.
    """
    validate_evaluation_class(cls)
    if _EVALUATION_REGISTRY.get(cls.name, cls) is not cls:
        raise PolicyError(colorstr("red", f"Evaluation name `{cls.name}` is already registered by "
                                          f"`{_EVALUATION_REGISTRY[cls.name].__qualname__}`."))
    _EVALUATION_REGISTRY[cls.name] = cls
    return cls



def get_evaluation_class(name: str) -> type:
    """
    The registered evaluation class named `name`.

    Args:
        name (str): The evaluation name.

    Raises:
        PolicyError: If no evaluation is registered under `name`.

    Returns:
        type: The evaluation class.
    """
    if name not in _EVALUATION_REGISTRY:
        raise PolicyError(colorstr("red", f"Unknown evaluation `{name}`; registered: {sorted(_EVALUATION_REGISTRY)}. "
                                          f"Use `file:<path.py>:<ClassName>` for an evaluation defined in a file."))
    return _EVALUATION_REGISTRY[name]



def registered_evaluations() -> dict:
    """
    Every evaluation class added with `register_evaluation` (there are no built-in evaluations).

    Returns:
        dict: ``{name: evaluation class}`` (a copy).
    """
    return dict(_EVALUATION_REGISTRY)

