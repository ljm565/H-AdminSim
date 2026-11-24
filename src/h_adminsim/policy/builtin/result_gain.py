from __future__ import annotations
from importlib import resources
from dataclasses import dataclass
from typing import ClassVar, Optional

from h_adminsim.utils import log, colorstr
from h_adminsim.registry.errors import PolicyError
from h_adminsim.policy.base import Policy, Decision
from h_adminsim.policy.context import ReviewContext



def _prompt(name: str) -> str:
    return str(resources.files("h_adminsim.assets.prompts").joinpath(name))



@dataclass
class ResultGainPolicy(Policy):
    """
    Staff-side negotiation policy (τ policy): how aggressively the administrative staff pushes a follow-up patient
    off their stated preference toward the hospital-preferred ``throughput_max`` schedule.

    Trigger index ``ti = GPC * TCL / τ`` (see ``NegotiationMetrics``), where τ is the per-preference trigger
    temperature. ``tcl_temperature`` and both τ are hospital-fixed constants; strength is set by
    ``negotiation_trigger_threshold``. Runs with different τ divide by their own τ, so anything comparing them
    recomputes ``GPC * TCL`` rather than trusting the stored ``ti``.

    ``mode`` selects when the staff negotiates:
        - ``'threshold'``     — negotiate once ``ti >= negotiation_trigger_threshold``.
        - ``'hospital-side'`` — always negotiates whenever there is anything to gain (fully hospital).
        - ``'patient-side'``  — never negotiates (fully patient).
    The two extremes decide without the trigger index (not via 0/inf arithmetic, which breaks under float division
    and the ``NegotiationMetrics`` temperature guard), so their trigger knobs are pinned to 1.0.

    Fields:
        mode: ``'threshold'``, ``'hospital-side'``, or ``'patient-side'``.
        negotiation_trigger_threshold: Cutoff on ``ti`` for the ``'threshold'`` mode (pinned to 1.0 on the extremes).
        tcl_temperature: Softmax temperature for TCL (hospital-fixed).
        trigger_temperature_visit: τ for ``visit_min`` patients (pinned to 1.0 on the extremes).
        trigger_temperature_stay: τ for ``stay_min`` patients (pinned to 1.0 on the extremes).
        negotiation_prompt_path: Staff persuasion prompt for the ``'threshold'`` mode (None -> the common prompt).
            The extremes always use their own (``'hospital-side'``: the hospital prompt, ``'patient-side'``: none).
    """
    name: ClassVar[str] = 'result_gain'
    MODES: ClassVar[tuple] = ('threshold', 'hospital-side', 'patient-side')
    TRIGGER_KNOBS: ClassVar[tuple] = ('trigger_temperature_visit', 'trigger_temperature_stay', 'negotiation_trigger_threshold')
    COMMON_PROMPT: ClassVar[str] = 'staff_negotiation_policy_common_system.txt'
    HOSPITAL_PROMPT: ClassVar[str] = 'staff_negotiation_policy_hospital_system.txt'

    mode: str = 'threshold'
    negotiation_trigger_threshold: float = 1.0
    tcl_temperature: float = 1.0
    trigger_temperature_visit: float = 1.0
    trigger_temperature_stay: float = 1.0
    negotiation_prompt_path: Optional[str] = None


    def __post_init__(self):
        """
        Check the parameters and set the negotiation prompt of the mode when the policy is built.

        Raises:
            PolicyError: If `mode` is unknown, or a trigger temperature (τ) is not positive.
        """
        if self.mode not in self.MODES:
            raise PolicyError(colorstr("red", f"Unknown negotiation mode `{self.mode}`; choose one of {list(self.MODES)}"))

        if self.mode == 'patient-side':
            self.negotiation_prompt_path = ''
            self._pin_trigger_knobs()
            log(f"Cleared negotiation prompt for '{self.mode}' policy (it never negotiates).", level='warning')
        elif self.mode == 'hospital-side':
            self.negotiation_prompt_path = _prompt(self.HOSPITAL_PROMPT)
            self._pin_trigger_knobs()
            log(f"Set negotiation prompt for '{self.mode}' policy.", level='info')
        elif not self.negotiation_prompt_path:
            self.negotiation_prompt_path = _prompt(self.COMMON_PROMPT)

        # τ divides the trigger index, so it must be positive (checked before any patient is simulated)
        for knob in ('trigger_temperature_visit', 'trigger_temperature_stay'):
            if getattr(self, knob) <= 0:
                raise PolicyError(colorstr("red", f"`{knob}` must be > 0 (it divides the trigger index GPC*TCL), got {getattr(self, knob)}."))


    @property
    def is_pole(self) -> bool:
        """Whether this is one of the extremes (``'hospital-side'`` / ``'patient-side'``)."""
        return self.mode != 'threshold'


    def _pin_trigger_knobs(self):
        """Force the trigger knobs to 1.0 on the extremes, which decide without ever reading them."""
        overridden = {k: getattr(self, k) for k in self.TRIGGER_KNOBS if getattr(self, k) != 1.0}
        for k in self.TRIGGER_KNOBS:
            setattr(self, k, 1.0)
        if overridden:
            log(f"'{self.mode}' policy ignores the trigger knobs; pinned to 1.0 so the recorded "
                f"ti stays unscaled (overrode {overridden}).", level='warning')


    def trigger_temperature_for(self, preference: str) -> float:
        """τ for the given preference: ``stay_min`` uses the stay temperature, everything else visit."""
        return self.trigger_temperature_stay if preference == 'stay_min' else self.trigger_temperature_visit


    def metrics_params(self, preference: str) -> dict:
        """The TCL softmax temperature; τ is applied later in ``trigger_index``."""
        return {'tcl_temperature': self.tcl_temperature}


    def trigger_index(self, metrics) -> float:
        """
        ``GPC * TCL / τ``; preferences without a τ (``throughput_max`` / ``indifferent``) use ``GPC * TCL``.
        A free win (``gpc == inf``) is ``inf`` regardless of TCL, so an idle device (TCL 0) cannot turn it into NaN.
        """
        if metrics['gpc'] == float('inf'):
            return float('inf')
        if metrics['preference'] in {'visit_min', 'stay_min'}:
            return metrics['gpc'] * metrics['tcl'] / self.trigger_temperature_for(metrics['preference'])
        return metrics['gpc'] * metrics['tcl']


    def negotiation_action(self, gpc: float, ti: Optional[float]) -> str:
        """
        ``'keep'`` (leave the patient on their preferred schedule), ``'auto'`` (a free win where the patient concedes
        nothing, ``gpc == inf``: book throughput_max without persuading) or ``'negotiate'`` (run the persuasion sub-loop).
        """
        # patient-side is the no-intervention baseline: never switch, not even a free win.
        if self.mode == 'patient-side':
            return 'keep'

        # Free win: switching costs the patient nothing and only helps, so just book it — no persuasion.
        if gpc == float('inf'):
            return 'auto'

        # hospital-side always persuades toward throughput_max whenever switching gains anything.
        if self.mode == 'hospital-side':
            return 'negotiate' if gpc > 0 else 'keep'

        # threshold negotiates only patients whose trigger index reaches the cutoff.
        return 'negotiate' if ti >= self.negotiation_trigger_threshold else 'keep'


    def review(self, ctx: ReviewContext) -> Decision:
        """
        Trigger index and decision, with the index computed once so the recorded ``ti`` is exactly the value the
        decision used (None on the extremes, which never read it).
        """
        ti = None if self.is_pole else self.trigger_index(ctx.metrics)
        action = self.negotiation_action(ctx.metrics['gpc'], ti)
        if action == 'negotiate':
            return Decision.negotiate(self.negotiation_prompt_path, ti=ti)
        if action == 'auto':
            return Decision.auto(ti=ti)
        return Decision.keep(ti=ti)
