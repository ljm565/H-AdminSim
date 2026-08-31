from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, field

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

    The staff negotiates a patient once their trigger index ``ti = PCI * TCL / trigger_temperature``
    (see ``NegotiationMetrics``) reaches ``negotiation_trigger_threshold``. ``tcl_temperature`` and the
    per-preference ``trigger_temperature_visit`` / ``trigger_temperature_stay`` are hospital-fixed
    constants shared across policies; negotiation strength is set per policy by
    ``negotiation_trigger_threshold`` alone — higher is stricter (fewer negotiations, patient-friendly),
    ``inf`` never negotiates (fully patient-side), ``0`` always negotiates (fully hospital-side).

    Fields:
        name: Identifier for the policy (e.g. 'full_patient', 'weak', 'strong', 'full_hospital').
        negotiation_prompt_path: Path to the staff persuasion system prompt used once a negotiation fires.
        tcl_temperature: Softmax temperature for TCL (hospital-fixed).
        trigger_temperature_visit: τ dividing the trigger index for ``visit_min`` patients (hospital-fixed).
        trigger_temperature_stay: τ dividing the trigger index for ``stay_min`` patients (hospital-fixed).
        negotiation_trigger_threshold: Per-policy cutoff on ``ti`` — the negotiation-strength knob.
    """
    name: str
    negotiation_prompt_path: str
    tcl_temperature: float = 1.0
    trigger_temperature_visit: float = 1.0
    trigger_temperature_stay: float = 1.0
    negotiation_trigger_threshold: float = 1.0



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
