"""Constants of the follow-up scheduling policy layer (`h_adminsim.policy`)."""

__all__ = [
    'POLICY_HOOKS',
    'POLICY_ACTIONS',
    'POLICY_SUPPORTED_OBJECTIVES',
    'POLICY_RESERVED_RECORD_KEYS',
    'POLICY_CONFIG_API_VERSION',
    'POLICY_CONFIG_KEYS',
    'POLICY_ENTRY_KEYS',
]



# Hook names a policy may implement. A policy implements a hook by overriding the method of that name.
POLICY_HOOKS = ('restrict', 'review')

# What a `review` decision may do with a patient whose proposed schedule differs from the hospital's
POLICY_ACTIONS = ('keep', 'auto', 'negotiate')

# Objectives the simulation engine can book after an `auto` / `negotiate` decision
POLICY_SUPPORTED_OBJECTIVES = ('throughput_max',)

# Fields of the per-booking `negotiation_metrics` record the simulation owns; a decision's `record` may not overwrite them
POLICY_RESERVED_RECORD_KEYS = frozenset({
    'preference', 'gpc', 'tcl', 'G', 'R', 'U', 'U_pref', 'U_thr', '_P', '_T',
    'negotiation_action', 'do_negotiate', 'decided_by', 'negotiation_outcome', 'negotiation_rounds',
})

# Policy config YAML format: the format version this code reads, and the keys allowed at the top level / per policy entry
POLICY_CONFIG_API_VERSION = 1
POLICY_CONFIG_KEYS = frozenset({'api_version', 'policies'})
POLICY_ENTRY_KEYS = frozenset({'id', 'source', 'params'})
