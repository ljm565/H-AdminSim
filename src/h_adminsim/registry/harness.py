# Severity of a harness `Issue`: an error means the policy cannot run as is, a warning that it runs but likely not as intended
HARNESS_ISSUE_LEVELS = ('error', 'warning')

# Directory generated policies are saved to by default (relative to the cwd)
HARNESS_DEFAULT_POLICY_DIR = 'policies'

# Subdirectory of the policy directory that keeps the previous version of every replaced file
HARNESS_BACKUP_DIR = '.backup'

# Text files a chat may save next to its policies (prompts and other assets declared under a config's `assets:`)
HARNESS_ASSET_EXTS = ('.txt', '.md')

# Per-chat policy folders: request headers carrying the chat id (in order), the folder of requests without one, and the
# marker file inside each folder that records its full chat id
HARNESS_CHAT_ID_HEADERS = ('X-Chat-Id', 'X-OpenWebUI-Chat-Id')
HARNESS_DEFAULT_CHAT_DIR = 'default'
HARNESS_CHAT_MARKER = '.chat_id'

# Smoke-test inputs, built from the agent data with the simulator's scheduler: cache file (in the policy directory),
# review samples per (preference, urgency, gain kind), most review samples, time budget for building them, how far
# ahead the first open date lies in the shifted restrict snapshot, and the search cap of each sample schedule
HARNESS_SAMPLES_FILE = 'samples.json'
HARNESS_SAMPLES_VERSION = 2    # bump when the way samples are built changes, so caches are rebuilt
HARNESS_SAMPLE_REVIEW_PER_KIND = 1
HARNESS_SAMPLE_REVIEW_MAX = 24
HARNESS_SAMPLE_TIME_BUDGET_S = 60.0
HARNESS_SAMPLE_SHIFT_LEAD_HOURS = 6.0
HARNESS_SAMPLE_SCHEDULE_BUDGET_S = 2.0    # search cap per schedule (stay_min can blow up)
HARNESS_SAMPLE_BUSY_LOADS = (0.5, 0.75, 0.9, 1.0)    # booked share of capacity per date in the busy snapshot

# Smoke test: patients per sample context, one per distinct combination of the patient fields the policies read
HARNESS_MAX_PATIENT_VARIANTS = 6

# `describe_data`: most distinct values listed one by one per patient field
HARNESS_DESCRIBE_MAX_VALUES = 10

# Negotiation prompts (the file of `Decision.negotiate(prompt_path)`, the staff agent's system prompt): the `str.format`
# keys the simulator fills (with the negotiation metrics, the deciding policy's record keys and the user metric ids; keep
# in sync with `OutpatientFollowUpScheduling._init_negotiation_agent`), the keys showing the two schedules, and the
# control tags that end a negotiation (accepted / given up, booking the preference / forced, booking the suggestion)
HARNESS_PROMPT_KEYS = (
    'dialogue_history', 'patient_preference', 'patient_preference_explanation', 'time_unit', '_P_summary', '_T_summary',
    'preference', 'gpc', 'tcl', 'G', 'R', 'U', 'U_pref', 'U_thr', '_P', '_T',
    'ti', 'negotiation_action', 'do_negotiate', 'decided_by',
)
HARNESS_PROMPT_SCHEDULE_KEYS = {'suggested': ('_T', '_T_summary'), 'preferred': ('_P', '_P_summary')}
HARNESS_PROMPT_ACCEPT_TAG = '#ACCEPT'
HARNESS_PROMPT_CLOSE_TAGS = ('#KEEP_PREFERENCE', '#FORCE_ACCEPT')

# Bundled staff negotiation prompts (under `h_adminsim/assets/prompts`) a chat can read as references
HARNESS_REFERENCE_PROMPTS = ('staff_negotiation_policy_common_system.txt', 'staff_negotiation_policy_hospital_system.txt')

# System prompt for an LLM writing policies with the harness tools (under `h_adminsim/assets/harness`)
HARNESS_SYSTEM_PROMPT = 'system_prompt.md'

# Tool server text the LLM reads (OpenAPI): the server description (the workflow) and the description of each tool
HARNESS_SERVER_DESCRIPTION = """Tools to write follow-up scheduling policies for the H-AdminSim hospital simulator.

Workflow: (1) read `get_policy_api` once before writing any policy; (2) check with `describe_data` that the data holds
what the policy needs, and look at `list_policies` / `read_policy` for existing policies; (3) start from
`make_policy_template` (and `make_metric_template` for a new metric); (4) write the code and `save_policy` it, which also validates
it; (5) fix every error the report lists and save again with overwrite=true until `ok` is true; (6) check the combined
policies with `validate_policy_config`. Never resolve a conflict between policies on your own: describe it and let the
user choose."""
HARNESS_TOOL_DESCRIPTIONS = {
    'get_policy_api': 'The policy API reference (rules, hooks, contexts, Decision, requires, config format, examples). Read it before writing a policy.',
    'describe_data': "What the server's agent data holds for the follow-up patients (each patient field: how many patients have it and which values occur) and what the simulator always provides whatever the data (built-in metrics, schedules, hospital state). Check it before planning: a rule that needs information neither listed here nor computable from it cannot run on this data.",
    'list_policies': "Policy files (.py), policy configs (.yaml) and text files available now, with whether each can be edited and its path (use the config paths in `schedule_task.policy_config`). Only this chat's own files are editable; the reference configs and the bundled negotiation prompts (`staff_negotiation_policy_*.txt`, to start your own prompt from) are read only. Other chats' policies are not shown: each chat works on its own. Call it again for every new request: files may have changed.",
    'read_policy': 'The content of a policy file, policy config or text file (e.g. a bundled negotiation prompt).',
    'make_policy_template': 'A runnable policy skeleton (Python code with TODOs) and the policy config that loads it.',
    'make_metric_template': "A user-defined metric skeleton (a `Metric` computing one value per negotiation from the review context, in its own `<name>_metric.py`) and the config listing it under `metrics:`. Use it when no built-in metric measures what a rule needs; save it with the policy that reads it in one `save_policy` call and one config.",
    'save_policy': "Save files to this chat's folder in one go and validate them together: the policy's and metrics' Python files (one class kind per file) and text assets such as a negotiation prompt (.txt, declared under the config's `assets:` and passed as `asset:<id>`), with the one config that loads them all. Fix every error in the report and save again with overwrite=true until `ok` is true. Replace a file you did not save in this conversation only after the user agreed; the previous version is kept in `.backup/`.",
    'validate_policy': 'Validate one policy file: syntax, class contract, parameters, purity, and a smoke test of its hooks on sample patients.',
    'validate_policy_config': 'Validate policy configs loaded together in priority order (as a simulation run would) and smoke-test the combined policies.',
}

# Scaffold templates (under `h_adminsim/assets/harness/templates`, `string.Template` placeholders): the policy file,
# the method of each hook, and the user-defined metric file
HARNESS_TEMPLATE_DIR = 'templates'
HARNESS_POLICY_TEMPLATE = 'policy.txt'
HARNESS_HOOK_TEMPLATES = {'restrict': 'restrict_hook.txt', 'review': 'review_hook.txt'}
HARNESS_METRIC_TEMPLATE = 'metric.txt'
