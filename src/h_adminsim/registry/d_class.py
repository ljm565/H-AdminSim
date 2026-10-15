from __future__ import annotations
import json
from importlib import resources
from dataclasses import dataclass, field
from typing import ClassVar, Optional, TYPE_CHECKING

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
    Base staff-side negotiation policy: how aggressively the administrative staff pushes a follow-up
    patient off their stated preference toward the hospital-preferred ``throughput_max`` schedule.

    Subclasses differ only in how the trigger index ``ti`` is computed (``trigger_index``) and which
    tests are booked at throughput_max before negotiating (``pinned_tests``); everything else lives here.

    ``mode`` selects when the staff negotiates, shared by every policy type:
        - ``'threshold'``     — tunable middle ground: negotiate once ``ti >= negotiation_trigger_threshold``.
        - ``'hospital-side'`` — always negotiates whenever there is anything to gain (fully hospital).
        - ``'patient-side'``  — never negotiates (fully patient).
    The two extremes are decided directly by ``negotiation_action`` (not via 0/inf arithmetic, which
    breaks under Python float division and the ``NegotiationMetrics`` ``trigger_temperature > 0``
    guard), so they behave identically across policy types. They never read the trigger knobs, which
    are therefore pinned to 1.0 (see ``_pin_trigger_knobs``).

    Fields:
        mode: ``'threshold'``, ``'hospital-side'``, or ``'patient-side'``.
        negotiation_prompt_path: Staff persuasion system prompt used once a negotiation fires (empty
            for ``'patient-side'``, which never negotiates).
        negotiation_trigger_threshold: Cutoff on ``ti`` for the ``'threshold'`` mode — the strength
            knob (higher = stricter, fewer negotiations; pinned to 1.0 on the extremes).
    """
    type: ClassVar[str] = ''
    MODES: ClassVar[tuple] = ('threshold', 'hospital-side', 'patient-side')
    TRIGGER_KNOBS: ClassVar[tuple] = ('negotiation_trigger_threshold',)

    mode: str = 'threshold'
    negotiation_prompt_path: str = str(resources.files("h_adminsim.assets.prompts").joinpath("staff_negotiation_policy_common_system.txt"))
    negotiation_trigger_threshold: float = 1.0


    def __post_init__(self):
        if self.mode not in self.MODES:
            raise ValueError(colorstr("red", f"Unknown negotiation mode `{self.mode}`; choose one of {list(self.MODES)}"))

        # patient-side never negotiates, so no persuasion prompt is ever used.
        if self.mode == 'patient-side':
            self.negotiation_prompt_path = ''
            self._pin_trigger_knobs()
            log(f"Cleared negotiation prompt for '{self.mode}' policy (it never negotiates).", level='warning')

        elif self.mode == 'hospital-side':
            self.negotiation_prompt_path = str(resources.files("h_adminsim.assets.prompts").joinpath("staff_negotiation_policy_hospital_system.txt"))
            self._pin_trigger_knobs()
            log(f"Set negotiation prompt for '{self.mode}' policy.", level='info')


    @property
    def is_pole(self) -> bool:
        """Whether this is one of the extremes (``'hospital-side'`` / ``'patient-side'``)."""
        return self.mode != 'threshold'


    def _pin_trigger_knobs(self):
        """
        Force the trigger knobs to 1.0 on the extremes, which decide without ever reading them.
        """
        overridden = {k: getattr(self, k) for k in self.TRIGGER_KNOBS if getattr(self, k) != 1.0}
        for k in self.TRIGGER_KNOBS:
            setattr(self, k, 1.0)
        if overridden:
            log(f"'{self.mode}' policy ignores the trigger knobs; pinned to 1.0 so the recorded "
                f"ti stays unscaled (overrode {overridden}).", level='warning')


    def metrics_kwargs(self, preference: str) -> dict:
        """
        Hospital-fixed constants this policy hands to ``NegotiationMetrics`` for a patient of `preference`
        (none by default; the trigger index itself is computed by ``trigger_index``).
        """
        return {}


    def pinned_tests(self, tests: list[dict]) -> list[dict]:
        """
        Tests booked at throughput_max before negotiating the rest. None by default.

        Args:
            tests (list[dict]): The patient's required tests (each carries at least `code`).

        Returns:
            list[dict]: The subset of `tests` to pin.
        """
        return []


    def trigger_index(self, metrics: dict) -> float:
        """
        Trigger index ``ti`` of a patient from their negotiation metrics; higher means more worth negotiating.

        Args:
            metrics (dict): Output of ``NegotiationMetrics.to_dict``.

        Returns:
            float: The trigger index (``inf`` for a free dominant win).
        """
        raise NotImplementedError(colorstr("red", f"'{type(self).__name__}' does not define a trigger index."))


    def decide(self, metrics: dict) -> dict:
        """
        Trigger index and negotiation decision for one patient, with the index computed once so the
        recorded ``ti`` is exactly the value the decision used.

        Args:
            metrics (dict): Output of ``NegotiationMetrics.to_dict``.

        Returns:
            dict: ``{'ti', 'negotiation_action', 'do_negotiate'}`` — ``ti`` is None on the extremes,
                  which never read it.
        """
        ti = None if self.is_pole else self.trigger_index(metrics)
        action = self.negotiation_action(metrics['gpc'], ti)
        return {'ti': ti, 'negotiation_action': action, 'do_negotiate': action == 'negotiate'}


    def negotiation_action(self, gpc: float, ti: Optional[float]) -> str:
        """
        Decide how to handle a patient whose staff-proposed schedule conflicts with throughput_max.

        Args:
            gpc (float): The patient's Gain Per Concession (``inf`` for a free win).
            ti (Optional[float]): The patient's trigger index; read by the ``'threshold'`` mode only.

        Returns:
            str: One of
                ``'keep'``      — leave the patient on their preferred schedule (no negotiation).
                ``'auto'``      — book throughput_max directly, without persuading: a free win where the
                                  patient concedes nothing (``G == 0`` -> ``gpc == inf``) yet gets results sooner.
                ``'negotiate'`` — run the persuasion sub-loop.
        """
        # patient-side is the no-intervention baseline: never switch, not even a free win.
        if self.mode == 'patient-side':
            return 'keep'
        
        # Free win: switching costs the patient nothing and only helps, so just book it — no persuasion.
        if gpc == float('inf'):
            return 'auto'

        # hospital-side always persuades toward throughput_max whenever switching gains anything.
        if self.mode == 'hospital-side':
            return 'negotiate' if gpc > 0 else 'keep'  # nothing to gain (gpc == 0) -> no-op

        # threshold negotiates only patients whose trigger index reaches the cutoff.
        return 'negotiate' if ti >= self.negotiation_trigger_threshold else 'keep'



@dataclass
class ResultGainNegotiationPolicy(StaffNegotiationPolicy):
    """
    Policy 1: trigger index ``ti = GPC * TCL / τ`` (see ``NegotiationMetrics``), where τ is the
    per-preference trigger temperature. ``tcl_temperature`` and both τ are hospital-fixed constants;
    strength is set by ``negotiation_trigger_threshold``. Runs with different τ divide by their own τ,
    so anything comparing them recomputes ``GPC * TCL`` rather than trusting the stored ``ti``.

    Fields:
        tcl_temperature: Softmax temperature for TCL (hospital-fixed).
        trigger_temperature_visit: τ for ``visit_min`` patients (pinned to 1.0 on the extremes).
        trigger_temperature_stay: τ for ``stay_min`` patients (pinned to 1.0 on the extremes).
    """
    type: ClassVar[str] = 'result_gain'
    TRIGGER_KNOBS: ClassVar[tuple] = ('trigger_temperature_visit', 'trigger_temperature_stay', 'negotiation_trigger_threshold')

    tcl_temperature: float = 1.0
    trigger_temperature_visit: float = 1.0
    trigger_temperature_stay: float = 1.0

    def __post_init__(self):
        super().__post_init__()

        # τ divides the trigger index, so it must be positive (checked here, before any patient is simulated)
        for knob in ('trigger_temperature_visit', 'trigger_temperature_stay'):
            if getattr(self, knob) <= 0:
                raise ValueError(colorstr("red", f"`{knob}` must be > 0 (it divides the trigger index GPC*TCL), got {getattr(self, knob)}."))


    def trigger_temperature_for(self, preference: str) -> float:
        """τ for the given preference: ``stay_min`` uses the stay temperature, everything else visit."""
        return self.trigger_temperature_stay if preference == 'stay_min' else self.trigger_temperature_visit


    def metrics_kwargs(self, preference: str) -> dict:
        """The TCL softmax temperature; τ is applied later in ``trigger_index``."""
        return {**super().metrics_kwargs(preference), 'tcl_temperature': self.tcl_temperature}


    def trigger_index(self, metrics: dict) -> float:
        """
        ``GPC * TCL / τ``; preferences without a τ (``throughput_max`` / ``indifferent``) use ``GPC * TCL``.
        """
        if metrics['preference'] in {'visit_min', 'stay_min'}:
            return metrics['gpc'] * metrics['tcl'] / self.trigger_temperature_for(metrics['preference'])
        return metrics['gpc'] * metrics['tcl']


NEGOTIATION_POLICY_TYPES = {cls.type: cls for cls in (ResultGainNegotiationPolicy)}
