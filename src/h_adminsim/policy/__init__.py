from .base import Policy, Decision
from h_adminsim.registry.errors import PolicyError, PolicyCompositionError, PolicyRequirementError
from h_adminsim.registry.policy import (
    POLICY_HOOKS,
    POLICY_ACTIONS,
    POLICY_SUPPORTED_OBJECTIVES,
    POLICY_CONFIG_API_VERSION,
)
from .context import RestrictContext, ReviewContext, PatientView, MetricView
from .stack import PolicyStack
from .metric import Metric
from .evaluation import Evaluation, RunData, run_evaluations
from .registry import (
    register_policy,
    register_metric,
    register_evaluation,
    validate_policy_class,
    validate_metric_class,
    validate_evaluation_class,
    get_policy_class,
    get_metric_class,
    get_evaluation_class,
    registered_policies,
    registered_metrics,
    registered_evaluations,
)
from .builder import (
    build_policy,
    build_metric,
    build_evaluation,
    default_policies,
    load_policy_config,
    snapshot_policy_config,
    pin_policy_config,
)
from .builtin import ResultGainPolicy, CapacityProtectionPolicy



__all__ = [
    # Policy API
    'Policy', 'Decision', 'Metric', 'Evaluation', 'RunData',
    'RestrictContext', 'ReviewContext', 'PatientView', 'MetricView',
    'PolicyStack',
    # Errors
    'PolicyError', 'PolicyCompositionError', 'PolicyRequirementError',
    # Constants
    'POLICY_HOOKS', 'POLICY_ACTIONS', 'POLICY_SUPPORTED_OBJECTIVES', 'POLICY_CONFIG_API_VERSION',
    # Registry
    'register_policy', 'register_metric', 'register_evaluation',
    'validate_policy_class', 'validate_metric_class', 'validate_evaluation_class',
    'get_policy_class', 'get_metric_class', 'get_evaluation_class',
    'registered_policies', 'registered_metrics', 'registered_evaluations',
    # Config loading
    'build_policy', 'build_metric', 'build_evaluation', 'default_policies',
    'load_policy_config', 'snapshot_policy_config', 'pin_policy_config',
    'run_evaluations',
    # Built-in policies
    'ResultGainPolicy', 'CapacityProtectionPolicy',
]
