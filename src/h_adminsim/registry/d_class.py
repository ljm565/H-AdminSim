from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from h_adminsim.utils import log, colorstr

if TYPE_CHECKING:
    from h_adminsim.agent import BaseAgent



@dataclass
class ConversationState:
    messages: list[dict] = field(default_factory=list)
    current_agent: str | None = None



@dataclass
class MASNode:
    """
    A single node in the Multi-Agent System tree.

    A node with no ``children`` is a leaf worker; otherwise its ``agent`` acts as
    a router that delegates to one child per turn.
    """
    name: str
    agent: BaseAgent
    children: dict[str, "MASNode"] = field(default_factory=dict)
    parent: Optional["MASNode"] = None
    description: Optional[str] = None
    is_complete: bool = False
    next_step: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return not self.children



@dataclass
class StaffNegotiationPolicy:
    """
    A staff-side negotiation policy: how aggressively the administrative staff pushes a follow-up
    patient off their stated preference toward the hospital-preferred ``throughput_max`` schedule.

    For the tunable ``'negotiation'`` policy the staff negotiates a patient once their trigger index
    ``ti = PCI * TCL / trigger_temperature`` (see ``NegotiationMetrics``) reaches
    ``negotiation_trigger_threshold``. ``tcl_temperature`` and the per-preference
    ``trigger_temperature_visit`` / ``trigger_temperature_stay`` are hospital-fixed constants; strength
    is set by ``negotiation_trigger_threshold`` (higher = stricter, fewer negotiations).

    The two extremes are decided directly by ``should_negotiate`` (not via 0/inf arithmetic, which
    breaks under Python float division and the ``NegotiationMetrics`` ``trigger_temperature > 0``
    guard): ``'patient-side'`` never negotiates (fully patient), ``'hospital-side'`` always negotiates
    (fully hospital). Both still carry finite temperatures so ``NegotiationMetrics`` stays valid.

    Fields:
        name: Policy identifier — ``'negotiation'`` (ti-thresholded), ``'patient-side'`` (never), or
            ``'hospital-side'`` (always).
        negotiation_prompt_path: Staff persuasion system prompt used once a negotiation fires (empty
            for ``'patient-side'``, which never negotiates).
        tcl_temperature: Softmax temperature for TCL (hospital-fixed).
        trigger_temperature_visit: τ dividing the trigger index for ``visit_min`` patients (hospital-fixed).
        trigger_temperature_stay: τ dividing the trigger index for ``stay_min`` patients (hospital-fixed).
        negotiation_trigger_threshold: Cutoff on ``ti`` for the ``'negotiation'`` policy — the strength knob.
    """
    name: str = 'negotiation'
    negotiation_prompt_path: str = ''
    tcl_temperature: float = 1.0
    trigger_temperature_visit: float = 1.0
    trigger_temperature_stay: float = 1.0
    negotiation_trigger_threshold: float = 1.0

    def __post_init__(self):
        if self.name not in ('negotiation', 'hospital-side', 'patient-side'):
            raise ValueError(colorstr("red", f"Unknown policy name: {self.name}"))

        # patient-side never negotiates, so no persuasion prompt is ever used.
        if self.name == 'patient-side' and self.negotiation_prompt_path:
            self.negotiation_prompt_path = ''
            log(f"Cleared negotiation prompt for '{self.name}' policy (it never negotiates).", level='warning')

    def trigger_temperature_for(self, preference: str) -> float:
        """τ for the given preference: ``stay_min`` uses the stay temperature, everything else visit."""
        return self.trigger_temperature_stay if preference == 'stay_min' else self.trigger_temperature_visit

    def should_negotiate(self, ti: float) -> bool:
        """Whether to negotiate this patient. Extremes short-circuit; ``'negotiation'`` thresholds ``ti``."""
        if self.name == 'patient-side':
            return False
        if self.name == 'hospital-side':
            return True
        return ti >= self.negotiation_trigger_threshold



@dataclass
class PatientNegotiationPolicy:
    """
    A patient-side negotiation policy: how the patient responds when the staff attempts to negotiate
    their follow-up test schedule toward ``throughput_max`` — i.e. how readily they concede their
    stated preference (``visit_min`` / ``stay_min``) versus hold out for it.

    Fields:
        name: Identifier for the policy (the patient's disposition toward conceding).
        negotiation_prompt_path: Path to the patient system prompt governing accept/refuse behavior
            during a negotiation.
    """
    name: str
    negotiation_prompt_path: str
