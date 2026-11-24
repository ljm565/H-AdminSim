__all__ = [
    'POLICY_HOOKS',
    'POLICY_ACTIONS',
    'POLICY_SUPPORTED_OBJECTIVES',
    'POLICY_RESERVED_RECORD_KEYS',
    'POLICY_CONFIG_API_VERSION',
    'POLICY_CONFIG_KEYS',
    'POLICY_ENTRY_KEYS',
    'POLICY_REQUIRES_PREFIX',
    'POLICY_PATIENT_SYMPTOM_FIELDS',
    'POLICY_METRIC_VALUE_TYPES',
    'POLICY_METRIC_RESERVED_IDS',
]



# Hook names a policy may implement. A policy implements a hook by overriding the method of that name.
POLICY_HOOKS = ('restrict', 'review')

# What a `review` decision may do with a patient whose proposed schedule differs from the hospital's
POLICY_ACTIONS = ('keep', 'auto', 'negotiate')

# Objectives the simulation engine can book after an `auto` / `negotiate` decision
POLICY_SUPPORTED_OBJECTIVES = ('throughput_max',)

# Reserved keys
POLICY_RESERVED_RECORD_KEYS = frozenset({
    'preference', 'gpc', 'tcl', 'G', 'R', 'U', 'U_pref', 'U_thr', '_P', '_T',
    'negotiation_action', 'do_negotiate', 'decided_by', 'negotiation_outcome', 'negotiation_rounds',
})

POLICY_METRIC_RESERVED_IDS = frozenset({'ti'})

# Policy config YAML format: the format version this code reads, the keys allowed at the top level (`metrics` and
# `evaluations` optional), and the keys allowed per policy / metric / evaluation entry
POLICY_CONFIG_API_VERSION = 1
POLICY_CONFIG_KEYS = frozenset({'api_version', 'policies', 'metrics', 'evaluations'})
POLICY_ENTRY_KEYS = frozenset({'id', 'source', 'params'})

# `requires` entries name patient fields as `patient.<field>`; the fields come from the agent data (see `patient_record`)
POLICY_REQUIRES_PREFIX = 'patient.'

# Fields of a follow-up test_data's `constraint.symptom` exposed on `ctx.patient` (agent-data key -> patient field)
POLICY_PATIENT_SYMPTOM_FIELDS = {'urgency': 'urgency', 'disease': 'disease', 'symptom': 'symptoms'}

# Values a user-defined metric may produce (stored in the booking's `negotiation_metrics` record and the result JSON)
POLICY_METRIC_VALUE_TYPES = (bool, int, float, str, type(None))
