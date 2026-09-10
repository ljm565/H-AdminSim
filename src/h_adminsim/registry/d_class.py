from __future__ import annotations
from importlib import resources
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

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

    The two extremes are decided directly by ``negotiation_action`` (not via 0/inf arithmetic, which
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
    negotiation_prompt_path: str = str(resources.files("h_adminsim.assets.prompts").joinpath("staff_negotiation_policy_common_system.txt"))
    tcl_temperature: float = 1.0
    trigger_temperature_visit: float = 1.0
    trigger_temperature_stay: float = 1.0
    negotiation_trigger_threshold: float = 1.0

    def __post_init__(self):
        if self.name not in ('negotiation', 'hospital-side', 'patient-side'):
            raise ValueError(colorstr("red", f"Unknown policy name: {self.name}"))

        # patient-side never negotiates, so no persuasion prompt is ever used.
        if self.name == 'patient-side':
            self.negotiation_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath("staff_negotiation_policy_patient_system.txt"))
            log(f"Cleared negotiation prompt for '{self.name}' policy (it never negotiates).", level='warning')
        
        elif self.name == 'hospital-side':
            self.negotiation_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath("staff_negotiation_policy_hospital_system.txt"))
            log(f"Set negotiation prompt for '{self.name}' policy.", level='info')

    
    def trigger_temperature_for(self, preference: str) -> float:
        """τ for the given preference: ``stay_min`` uses the stay temperature, everything else visit."""
        return self.trigger_temperature_stay if preference == 'stay_min' else self.trigger_temperature_visit

    
    def negotiation_action(self, pci: float, ti: float) -> str:
        """
        Decide how to handle a patient whose staff-proposed schedule conflicts with throughput_max.

        Returns one of:
            ``'keep'``      — leave the patient on their preferred schedule (no negotiation).
            ``'auto'``      — book throughput_max directly, without persuading: a free win where the
                              patient concedes nothing (``G == 0`` -> ``pci == inf``) yet gets results sooner.
            ``'negotiate'`` — run the persuasion sub-loop.

        Extremes short-circuit (``'patient-side'`` never negotiates, ``'hospital-side'`` always does when
        there is anything to gain); ``'negotiation'`` thresholds ``ti``.
        """
        # Free win: switching costs the patient nothing and only helps, so just book it — no persuasion.
        if pci == float('inf') and self.name != 'patient-side':
            return 'auto'
        if self.name == 'patient-side':
            return 'keep'
        if self.name == 'hospital-side':
            return 'negotiate' if pci > 0 else 'keep'  # nothing to gain (pci == 0) -> no-op
        return 'negotiate' if ti >= self.negotiation_trigger_threshold else 'keep'



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
