import os
import re
import string
from typing import Iterable

from h_adminsim.registry.harness import (
    HARNESS_PROMPT_KEYS,
    HARNESS_PROMPT_SCHEDULE_KEYS,
    HARNESS_PROMPT_ACCEPT_TAG,
    HARNESS_PROMPT_CLOSE_TAGS,
    HARNESS_REFERENCE_PROMPTS,
)
from h_adminsim.utils.filesys_utils import bundled_prompt



def bundled_prompt_files() -> list[str]:
    """
    The staff negotiation prompts shipped with the package, which a chat can read as references for its own prompts.

    Returns:
        list[str]: Their paths.
    """
    paths = [bundled_prompt(name) for name in HARNESS_REFERENCE_PROMPTS]
    return [path for path in paths if os.path.isfile(path)]



def prompt_contract() -> list[str]:
    """
    What a negotiation prompt must contain, for the policy API description.

    Returns:
        list[str]: One rule per item (markdown).
    """
    keys = ', '.join(f'`{{{k}}}`' for k in HARNESS_PROMPT_KEYS)
    close = ' or '.join(f'`{t}`' for t in HARNESS_PROMPT_CLOSE_TAGS)
    return [
        "`Decision.negotiate(prompt_path)` names the staff agent's system prompt for the persuasion dialogue. The "
        "simulator fills it with `str.format` when the negotiation starts.",
        f"Placeholders it fills: {keys}, the user metric ids, and the keys of the deciding decision's `record`. "
        "`{_P_summary}` / `{_T_summary}` describe the patient's preferred and the suggested schedule in words, `{_P}` / "
        "`{_T}` give their raw slots, and `{dialogue_history}` is the conversation so far. Any other `{name}` raises "
        "KeyError, so write literal braces as `{{` and `}}`.",
        f"The staff ends its message with a control tag: `{HARNESS_PROMPT_ACCEPT_TAG}` once the patient agrees (the "
        f"suggested schedule is booked), and {close} when the patient keeps refusing (`#KEEP_PREFERENCE` books the "
        "patient's preference, `#FORCE_ACCEPT` the suggestion anyway). The prompt must tell the staff when to use them; "
        "without them a negotiation never closes and fails at the turn limit.",
        f"Start from a bundled prompt ({', '.join(f'`{n}`' for n in HARNESS_REFERENCE_PROMPTS)}; the first gives up with "
        "`#KEEP_PREFERENCE`, the second forces with `#FORCE_ACCEPT`): keep its placeholders and closing rules, and "
        "change the tone and arguments. The validation checks every prompt a policy negotiates with against these rules.",
    ]



def check_negotiation_prompt(path: str, extra_keys: Iterable[str] = (), strict: bool = True) -> list[tuple]:
    """
    Check a negotiation prompt against what the simulator does with it: fill it with `str.format`, and end the
    negotiation on a control tag.

    Args:
        path (str): The prompt file.
        extra_keys (Iterable[str], optional): Other keys the simulator fills for it (user metric ids, the deciding
                                              decision's record keys). Defaults to ().
        strict (bool, optional): Whether an unknown placeholder is an error; False when the record keys are not known
                                 (a policy that did not negotiate on any sample), which makes it a warning.
                                 Defaults to True.

    Returns:
        list[tuple]: ``(level, code, message)`` per problem; empty if the prompt is fine.
    """
    name = os.path.basename(path)
    try:
        with open(path) as f:
            text = f.read()
    except OSError as e:
        return [('error', 'prompt_unreadable', f"Could not read the negotiation prompt `{name}`: {e}")]
    try:
        fields = [f for _, f, _, _ in string.Formatter().parse(text) if f is not None]
    except ValueError as e:
        return [('error', 'prompt_format', f"`{name}` is not a valid `str.format` template ({e}); write literal braces "
                                           f"as `{{{{` and `}}}}`.")]

    issues = []
    names = {re.split(r'[.\[]', f, maxsplit=1)[0] for f in fields}
    if any(n == '' or n.isdigit() for n in names):
        issues.append(('error', 'prompt_format', f"`{name}` has positional placeholders (`{{}}` or `{{0}}`); the "
                                                 f"simulator fills placeholders by name. Write literal braces as `{{{{` "
                                                 f"and `}}}}`."))
    known = set(HARNESS_PROMPT_KEYS) | set(extra_keys)
    unknown = sorted(n for n in names if n and not n.isdigit() and n not in known)
    if unknown:
        what = (f"`{name}` uses placeholders the simulator does not fill: {unknown}. The negotiation would fail with "
                f"KeyError; use the documented keys, or write literal braces as `{{{{` and `}}}}`.")
        if not strict:
            what += " (Unless the policy's decision `record` sets them: it negotiated on no sample, so its record is unknown.)"
        issues.append(('error' if strict else 'warning', 'prompt_unknown_key', what))
    if HARNESS_PROMPT_ACCEPT_TAG not in text:
        issues.append(('error', 'prompt_no_accept_tag',
                       f"`{name}` never tells the staff to end with `{HARNESS_PROMPT_ACCEPT_TAG}` once the patient agrees, "
                       f"so an agreed negotiation never closes and fails at the turn limit. Add that instruction, as in "
                       f"the bundled prompts."))
    if not any(tag in text for tag in HARNESS_PROMPT_CLOSE_TAGS):
        issues.append(('error', 'prompt_no_close_tag',
                       f"`{name}` has no way to end when the patient keeps refusing: tell the staff to end with "
                       f"`#KEEP_PREFERENCE` (book the patient's preference) or `#FORCE_ACCEPT` (book the suggestion "
                       f"anyway). Otherwise the negotiation fails at the turn limit."))
    for which, alternatives in HARNESS_PROMPT_SCHEDULE_KEYS.items():
        if not names & set(alternatives):
            issues.append(('warning', 'prompt_missing_key',
                           f"`{name}` never shows the {which} schedule ({' or '.join(f'`{{{k}}}`' for k in alternatives)}), "
                           f"so the staff cannot tell the patient what it is."))
    if 'dialogue_history' not in names:
        issues.append(('warning', 'prompt_missing_key',
                       f"`{name}` does not include `{{dialogue_history}}`, the conversation before the negotiation, so the "
                       f"staff does not know what the patient already said."))
    return issues
