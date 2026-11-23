from __future__ import annotations

from h_adminsim.utils import colorstr
from h_adminsim.registry.errors import PolicyError
from h_adminsim.policy.base import Policy
from h_adminsim.policy.builtin import ResultGainPolicy, CapacityProtectionPolicy



_REGISTRY: dict = {cls.name: cls for cls in (ResultGainPolicy, CapacityProtectionPolicy)}


def validate_policy_class(cls) -> type:
    """
    Check that `cls` can serve as a policy.

    Raises:
        PolicyError: If `cls` is not a `Policy` subclass, has no `name`, or implements no hook.
    """
    if not (isinstance(cls, type) and issubclass(cls, Policy)):
        raise PolicyError(colorstr("red", f"`{cls}` is not a Policy subclass."))
    if not cls.name:
        raise PolicyError(colorstr("red", f"Policy class `{cls.__qualname__}` needs a non-empty `name`."))
    if not cls.hooks():
        raise PolicyError(colorstr("red", f"Policy `{cls.name}` implements no hook (`restrict` / `review`)."))
    return cls


def register_policy(cls: type) -> type:
    """
    Register a policy class under its `name` so `builtin:<name>` sources can refer to it. Usable as a decorator.

    Raises:
        PolicyError: If `cls` is not a valid policy class, or the name is taken by another class.
    """
    validate_policy_class(cls)
    if _REGISTRY.get(cls.name, cls) is not cls:
        raise PolicyError(colorstr("red", f"Policy name `{cls.name}` is already registered by `{_REGISTRY[cls.name].__qualname__}`."))
    _REGISTRY[cls.name] = cls
    return cls


def get_policy_class(name: str) -> type:
    if name not in _REGISTRY:
        raise PolicyError(colorstr("red", f"Unknown policy `{name}`; choose one of {sorted(_REGISTRY)}."))
    return _REGISTRY[name]


def registered_policies() -> dict:
    return dict(_REGISTRY)
