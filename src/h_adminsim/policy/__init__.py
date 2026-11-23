"""
Follow-up scheduling policies.

The simulation calls policies only through `PolicyStack` hooks (`restrict`, `review`), handing them read-only
snapshots (`RestrictContext`, `ReviewContext`). Built-in policies: `result_gain` (τ negotiation trigger) and
`capacity_protection` (urgent capacity protection). A run's policies come from a policy config YAML
(`load_policy_config`); custom policies subclass `Policy` and are referenced there as `file:<path.py>:<ClassName>`
(or registered with `register_policy` and referenced as `builtin:<name>`).
"""
from .base import Policy, Decision
from h_adminsim.registry.errors import PolicyError, PolicyCompositionError
from h_adminsim.registry.policy import (
    POLICY_HOOKS,
    POLICY_ACTIONS,
    POLICY_SUPPORTED_OBJECTIVES,
    POLICY_CONFIG_API_VERSION,
)
from .context import RestrictContext, ReviewContext
from .stack import PolicyStack
from .registry import register_policy, validate_policy_class, get_policy_class, registered_policies
from .builder import (
    build_policy,
    default_policies,
    load_policy_config,
    snapshot_policy_config,
    pin_policy_config,
)
from .metadata import run_policies, run_policy_summary
from .builtin import ResultGainPolicy, CapacityProtectionPolicy
