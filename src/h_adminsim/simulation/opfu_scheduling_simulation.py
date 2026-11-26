import re
import json
import random
from copy import deepcopy
from dataclasses import replace
from collections import defaultdict
from patientsim import PatientAgent
from decimal import Decimal, getcontext
from langchain.agents import AgentExecutor
from typing import Union, Callable, Iterator, Optional, TYPE_CHECKING

from h_adminsim.registry.errors import (
    STATUS_CODES,
    SchedulingError,
    ToolCallingError,
    DataNotFoundError,
    ToolSelectionError,
    AgentSelectionError,
)
from h_adminsim.registry.variables import SCHEDULE_STATUS
from h_adminsim.registry.phrases import (
    OPFU_PREFERENCE_PHRASE_STAFF,
    OPFU_PREFERENCE_PHRASE_PATIENT,
    OPFU_PROPOSAL_PHRASE_STAFF,
    OPFU_UNAVAILABLE_PHRASE_STAFF,
)
from h_adminsim.tools.callback import TokenUsageCallback
from h_adminsim.tools.sanity_checker import SanityChecker
from h_adminsim.tools.scheduling_rule import SchedulingRule, scheduling_tool_calling
from h_adminsim.tools.negotiation_metrics import NegotiationMetrics
from h_adminsim.policy.stack import PolicyStack
from h_adminsim.policy.context import RestrictContext, ReviewContext
from h_adminsim.policy.builder import default_policies
from h_adminsim.simulation.op_scheduling_simulation import OPSchedulingSimulation, TurnLimitReached
from h_adminsim.utils import log, colorstr, Information
from h_adminsim.utils.common_utils import (    
    iso_to_date,
    iso_to_hour,
    get_iso_time,
    hour_to_hhmmss,
    add_hours_to_iso,
    compare_iso_time,
    init_result_dict,
    preprocess_dialog,
    calculate_idle_wait,
    describe_unavailable,
    render_availability_tables,
)

if TYPE_CHECKING:
    from h_adminsim.pipeline.h_mas import HospitalMAS
    from h_adminsim.agent.scheduling_admin_staff import SchedulingAdminStaffAgent
    from h_adminsim.environment.hospital import HospitalEnvironment




class OPFUSchedulingSimulation(OPSchedulingSimulation):
    HISTORY_KEYS = ('test_scheduling', 'test_cancel', 'test_reschedule')
    REJECTION_PROMPT = 'opfu_schedule_patient_rejected_system.txt'
    NOT_FOUND_MESSAGE = "Sorry, we couldn't find your scheduled tests. Could you please check your details again (patient and doctor names)?"

    # Per-round wrappers that hand each negotiation agent the current round + the other party's last line
    STAFF_NEGOTIATION_TURN = (
        "[Negotiation round {round}. If your closing rules say this is where you stop negotiating, close the "
        "conversation now and end your message with the exact control tag those rules name for that close.]\n\n"
        'This is the patient\'s response: "{utterance}"\n\n'
        "Reply now, staying strictly within all of your negotiation rules above."
    )
    PATIENT_NEGOTIATION_TURN = (
        "[Negotiation round {round}: the staff keeps pressing you to change your schedule.]\n\n"
        'This is the staff\'s response: "{utterance}"\n\n'
        "Reply now, staying strictly within your negotiation behavior and persona above."
    )

    # Lead-in for the closing confirmation
    NEGOTIATION_CLOSING_LEAD = {
        'accepted': "Great — here's how I'll arrange it, then.",
        'forced': "Since getting your results back early matters for your care, I'll go ahead and arrange it this way.",
        'kept': "Of course — I'll book it the way you asked, then.",
    }

    # Negotiation outcome -> the patient's closing persona. 
    # `auto` concedes nothing and `kept` books the patient's own preference after the staff gave up, so both close satisfied.
    NEGOTIATION_CLOSINGS = {
        'none': 'satisfied',
        'auto': 'satisfied',
        'kept': 'satisfied',
        'accepted': 'conceded',
        'forced': 'imposed',
    }

    def __init__(self,
                 patient_agent: PatientAgent,
                 admin_staff_mas: "HospitalMAS",
                 metadata: dict,
                 department_data: dict,
                 environment: "HospitalEnvironment",
                 scheduling_strategy: str = 'tool_calling',
                 preference_rejection_prob: float = 0.3,
                 preference_rejection_prob_decay: float = 0.5,
                 fhir_integration: bool = False,
                 schedule_rejection_prompt_path: Optional[str] = None,
                 sanity_checker: Optional[SanityChecker] = None,
                 policies: Optional[PolicyStack] = None,
                 patient_records: Optional[dict] = None):
        """
        Set up the follow-up test-scheduling simulation.

        Args:
            patient_agent (PatientAgent): The simulated patient.
            admin_staff_mas (HospitalMAS): The administration staff multi-agent system.
            metadata (dict): Hospital metadata (``time`` and ``days``).
            department_data (dict): Department information of the hospital.
            environment (HospitalEnvironment): The hospital environment.
            scheduling_strategy (str, optional): `'tool_calling'` or `'reasoning'`. Defaults to 'tool_calling'.
            preference_rejection_prob (float, optional): Initial probability the patient rejects a proposal. Defaults to 0.3.
            preference_rejection_prob_decay (float, optional): Factor applied to that probability after each rejection. Defaults to 0.5.
            fhir_integration (bool, optional): Whether to use the FHIR server. Defaults to False.
            schedule_rejection_prompt_path (Optional[str], optional): Custom rejection system prompt file. Defaults to None.
            sanity_checker (Optional[SanityChecker], optional): Checker grading predicted schedules; None skips grading. Defaults to None.
            policies (Optional[PolicyStack], optional): The run's scheduling policies; None uses `default_policies()`. Defaults to None.
            patient_records (Optional[dict], optional): ``{patient name: agent-data record}`` exposed to policies as `ctx.patient`. Defaults to None.
        """
        super().__init__(patient_agent, admin_staff_mas)

        # Initialize simulation parameters
        getcontext().prec = 10
        self._chief_agent_name = 'follow_up_visit_scheduling'
        self.environment = environment
        self._START_HOUR = metadata['time']['start_hour']
        self._END_HOUR = metadata['time']['end_hour']
        self._TIME_UNIT = metadata['time']['interval_hour']
        self._DAY = metadata['days']
        self.scheduling_strategy = scheduling_strategy
        self.preference_rejection_prob = preference_rejection_prob
        self.preference_rejection_prob_decay = preference_rejection_prob_decay
        self.fhir_integration = fhir_integration
        self._init_prompt(schedule_rejection_prompt_path)
        self.sanity_checker = sanity_checker
        self.rules = SchedulingRule(metadata, department_data, self.environment, self.fhir_integration)
        self.policies = policies if policies is not None else default_policies()   # the run's scheduling policies (see `h_adminsim.policy`)
        self._last_decision = None       # latest `review` decision; its prompt drives the persuasion agent
        self._restrictions = {}          # {restrict policy id: blocked (test_code, date) pairs} for the current patient
        self.patient_records = patient_records or {}   # {patient name: agent-data record} exposed to policies as `ctx.patient`
        self._negotiation_agent = None   # standalone persuasion agent, built lazily on first negotiation

    
    def _rejection_prompt_fields(self,
                                 patient_condition: dict,
                                 rejected_preference: str) -> dict:
        """
        Phrase a follow-up test-scheduling preference for the schedule-rejection prompt.

        Follow-up preferences name an objective (no doctor/date to fill); `occupation` / `distance` come from `patient_condition`.

        Args:
            patient_condition (dict): Patient ground-truth condition including current preference,
                                      occupation, hospital distance, and test-time unavailability.
            rejected_preference (str): The preference the staff agent proposed in the previous turn.

        Returns:
            dict: Keyword arguments for `rejection_system_prompt_template.format`.
        """
        preference = patient_condition.get('preference')
        return {
            'preference': preference,
            'preference_desc': OPFU_PREFERENCE_PHRASE_PATIENT[preference],
            'rejected_preference': OPFU_PROPOSAL_PHRASE_STAFF[rejected_preference],
            'occupation': patient_condition.get('occupation'),
            'distance': patient_condition.get('distance'),
            'unavailable': describe_unavailable(patient_condition.get('unavailable')),
        }


    @property
    def scheduling_agent(self) -> "SchedulingAdminStaffAgent":
        """
        The ``'follow_up_visit_scheduling'`` worker, fetched from the MAS on each access.
        """
        return self.admin_staff_mas.get_agent('follow_up_visit_scheduling')


    @staticmethod
    def _summarize_schedule_for_negotiation(schedule: dict) -> str:
        """
        Plain-language summary of one candidate schedule for the persuasion prompt.

        Pre-renders visit days and finish time so the persuasion agent need not derive them from raw slots.

        Args:
            schedule (dict): A `pred_schedule`-shaped schedule (`test_schedule` as a start-sorted list
                             with hour-float `start` / `end`), i.e. the metrics' `_P` or `_T`.

        Returns:
            str: Visit days with clock times, the result-ready moment, and same-day idle waiting.
        """
        tests = schedule.get('test_schedule') or []
        if not tests:
            return 'No test could be placed.'

        def _clock(hour: Union[str, float]) -> str:
            return hour_to_hhmmss(float(hour))[:5]

        by_date = defaultdict(list)
        for test in sorted(tests, key=lambda t: (t['date'], t['start'])):
            by_date[test['date']].append(f"{test['name']} {_clock(test['start'])}-{_clock(test['end'])}")

        days = '; '.join(f"{date} ({', '.join(items)})" for date, items in by_date.items())
        ready = schedule.get('all_results_ready_at')
        ready_text = f"{iso_to_date(ready)} {_clock(iso_to_hour(ready))}" if ready else 'unknown'
        idle = schedule.get('idle_waiting_time') or 0
        return (f"{len(by_date)} visit day(s) — {days}. "
                f"All results ready by {ready_text}. "
                f"Idle waiting between same-day tests: {idle} hour(s).")


    def init_negotiation_agent(self,
                               patient_preference: str,
                               negotiation_metrics: dict) -> Optional["SchedulingAdminStaffAgent"]:
        """
        Standalone persuasion agent used to negotiate the patient toward ``throughput_max``.

        Built lazily on first use with the staff's model but its own client (separate token usage), no MAS
        routing or tools, and the prompt of the `review` policy's ``'negotiate'`` decision.

        Args:
            patient_preference (str): The patient's stated preference for scheduling (e.g., 'visit_min', 'stay_min').
            negotiation_metrics (dict): Metrics computed by `_calculate_negotiation_metrics`, including `gpc`, `tcl`, `ti`, and other relevant values.

        Returns:
            Optional[SchedulingAdminStaffAgent]: The negotiation agent, or None if no negotiation was decided.
        """
        if self._last_decision is None or not self._last_decision.do_negotiate:
            return None
        if self._negotiation_agent is None:
            from h_adminsim.agent.scheduling_admin_staff import SchedulingAdminStaffAgent
            staff = self.scheduling_agent
            is_managed = ('gpt' in staff.model.lower()) or ('gemini' in staff.model.lower())
            self._negotiation_agent = SchedulingAdminStaffAgent(
                target_task='follow_up_visit_scheduling',
                model=staff.model,
                use_vllm=not is_managed,
                vllm_endpoint=getattr(staff.client, 'vllm_endpoint', None),
                system_prompt_path=self._last_decision.prompt_path,
                log_verbose=False,
                temperature=0 if not 'gpt-5' in staff.model.lower() else 1,
                prompt_kwargs={
                    'system_prompt': {
                        'patient_preference': patient_preference,
                        'patient_preference_explanation': OPFU_PREFERENCE_PHRASE_STAFF[patient_preference],
                        'dialogue_history': preprocess_dialog(self.dialog_history['test_scheduling'][:-1]),
                        'time_unit': self._TIME_UNIT,
                        **negotiation_metrics,
                        '_P_summary': self._summarize_schedule_for_negotiation(negotiation_metrics['_P']),
                        '_T_summary': self._summarize_schedule_for_negotiation(negotiation_metrics['_T']),
                    },
                },
            )
        return self._negotiation_agent


    def _init_negotiation_metrics(self, preference: Optional[str] = None) -> dict:
        """
        Initialize negotiation metrics with default values.

        Args:
            preference (Optional[str], optional): The scheduling preference to initialize metrics for. Defaults to None.

        Returns:
            dict: Same keys as `_calculate_negotiation_metrics` (plus user metrics), all empty; outcome 'none'.
        """
        # Same shape as `_calculate_negotiation_metrics` (NegotiationMetrics.to_dict + policy/outcome fields),
        # so a patient with no computed metrics still exports a consistent record.
        return {
            'preference': preference,
            'gpc': None,
            'tcl': None,
            'ti': None,
            'do_negotiate': False,
            'G': None,
            'R': None,
            'U': None,
            'U_pref': None,
            'U_thr': None,
            '_P': None,
            '_T': None,
            'negotiation_action': None,
            'decided_by': None,              # id of the policy that made `negotiation_action`
            'negotiation_outcome': 'none',   # 'accepted' | 'forced' | 'kept' | 'auto' | 'none' (booked _T on accepted/forced/auto)
            'negotiation_rounds': 0,         # number of staff persuasion pitches
            **{m.id: None for m in self.policies.metrics},   # user-defined metrics (see `h_adminsim.policy.Metric`)
        }

    
    def _render_staff_reply(self,
                            prediction: dict,
                            reply_type: str,
                            gt_patient_condition: Optional[dict] = None,
                            staff_known_data: Optional[dict] = None,
                            natural_express: bool = True,
                            prev_suggested_schedule: Optional[dict] = None) -> str:
        """
        Turn a structured staff scheduling result into a natural-language utterance.

        Args:
            prediction (dict): The result of `scheduling`, either a clarification (``type == 'text'``) or a schedule proposal (``type == 'tool'``).
            reply_type (str): Reply types of the scheduling agent.
            gt_patient_condition (Optional[dict], optional): Ground-truth condition for this turn; its required tests
                                                              flag unscheduled ones (needed for schedule proposals). Defaults to None.
            staff_known_data (Optional[dict], optional): Patient information known to the staff; supplies the follow-up
                                                          doctor (needed for schedule proposals). Defaults to None.
            natural_express (bool, optional): Whether to phrase a schedule proposal naturally instead of dumping the raw schedule dict. Defaults to True.
            prev_suggested_schedule (Optional[dict], optional): The previous (rejected) staff suggestion. Defaults to None.

        Returns:
            str: The staff utterance to show the patient (falls through to None for an unrecognized result).
        """
        # Clarification message
        if prediction['type'] == 'text':
            return prediction['result']
        
        elif prediction['type'] == 'tool':
            # Test cancellation confirmation
            if reply_type == 'test_cancel':
                result = prediction['result']

                # A wrong identification cancels nothing; the loop surfaces it as a failure.
                if result['cancelled_schedule'] is None:
                    return ""

                # Successful cancellation
                cancelled = {k: v for k, v in result['cancelled_schedule'].items()
                             if k in ['patient', 'attending_physician', 'department', 'date', 'test']}
                return f"I've cancelled all your scheduled tests: {cancelled}"

            # Test rescheduling confirmation
            if reply_type == 'test_reschedule':
                tmp_flag = prediction.get('tmp_flag')
                result = prediction['result']

                # Failure cases have nothing to confirm; the loop surfaces them as raises.
                if tmp_flag not in ('waiting_list', 'reschedule'):
                    return ""

                original = {k: v for k, v in result['original_schedule'].items()
                            if k in ['patient', 'attending_physician', 'department', 'date', 'test']}

                # No earlier slot available -> added to the waiting list
                if tmp_flag == 'waiting_list':
                    return f"There are no earlier times right now. I've added your tests to the waiting list: {original}"

                # Successfully moved every test earlier
                new = {k: v for k, v in result['new_schedule'].items()
                       if k in ['patient', 'attending_physician', 'department', 'date', 'test']}
                return f"I've moved your tests earlier: {new}"

            if 'test_list' in prediction['result']:
                test_list = prediction['result']['test_list']
                test_list_desc = ', '.join([t['name'] for t in test_list])

                # Response formatting for test guidance
                try:
                    if natural_express:
                        _format = random.choice(self.scheduling_agent.natural_test_explanation) \
                            if isinstance(self.scheduling_agent.natural_test_explanation, list) \
                                else self.scheduling_agent.natural_test_explanation
                        return _format.format(
                            test_len=len(test_list),
                            test_list=test_list_desc
                        ) + ' ' + self.scheduling_agent.test_greet
                    else:
                        return self.scheduling_agent.test_explanation.format(
                            test_len=len(test_list),
                            test_list=test_list_desc
                        ) + ' ' + self.scheduling_agent.test_greet
                except:
                    try:
                        return self.scheduling_agent.test_explanation.format(
                            test_len=len(test_list),
                            test_list=test_list_desc
                        ) + ' ' + self.scheduling_agent.test_greet
                    except:
                        return 'Your tests: ' + str(test_list_desc) + ' ' + self.scheduling_agent.test_greet
            
            elif 'test_schedule' in prediction['result']:
                pred_schedule  = prediction['result']
                pred_test_schedules = pred_schedule['test_schedule']

                # Build a humanized summary of every test slot
                parts = []
                for test_info in pred_test_schedules:
                    parts.append(
                        f"{test_info['name']} on {test_info['date']} from {test_info['start']} to {test_info['end']}"
                    )

                # Notify the patient of any required tests that the agent could not
                # fit within the simulation window (no deferred booking is attempted).
                required_test_codes = {t['code'] for t in gt_patient_condition.get('test', [])}
                unscheduled_tests = {test_info['name'] for test_info in pred_test_schedules if test_info['code'] not in required_test_codes}
                if unscheduled_tests:
                    parts.append(
                        f"however, the scheduling for {', '.join(sorted(unscheduled_tests))} test(s) will be arranged later"
                    )

                fu_slot = pred_schedule['fu_schedule']
                fu_doctor = staff_known_data['patient_fv']['attending_physician']
                if isinstance(fu_slot, dict) and fu_slot:
                    fu_info = fu_slot[fu_doctor]
                    parts.append(
                        f"follow-up with {fu_doctor} on {fu_info['date']} from {fu_info['start']} to {fu_info['end']}"
                    )
                else:
                    all_results_ready_at = pred_schedule.get('all_results_ready_at')
                    if all_results_ready_at:
                        parts.append(
                            f"and can I make a follow-up appointment with {fu_doctor} after {all_results_ready_at}"
                        )
                    else:
                        parts.append(f"(follow-up with {fu_doctor} cannot be booked at this time)")
                
                schedule_summary = "; ".join(parts)

                # Same schedule as the previous (rejected) suggestion -> say so instead of re-proposing it
                is_same_as_prev = prev_suggested_schedule is not None and \
                    self._schedule_signature(pred_schedule) == self._schedule_signature(prev_suggested_schedule)
                if is_same_as_prev:
                    try:
                        if natural_express:
                            _format = random.choice(self.scheduling_agent.natural_same_fu_schedule_suggestion) \
                                if isinstance(self.scheduling_agent.natural_same_fu_schedule_suggestion, list) \
                                    else self.scheduling_agent.natural_same_fu_schedule_suggestion
                            return _format.format(schedule_summary=schedule_summary)
                        else:
                            return self.scheduling_agent.same_fu_schedule_suggestion.format(
                                schedule_summary=schedule_summary
                            )
                    except:
                        return 'Same test schedules as before: ' + schedule_summary

                # Response formatting for test guidance
                try:
                    if natural_express:
                        _format = random.choice(self.scheduling_agent.natural_fu_schedule_suggestion) \
                            if isinstance(self.scheduling_agent.natural_fu_schedule_suggestion, list) \
                                else self.scheduling_agent.natural_fu_schedule_suggestion
                        return _format.format(schedule_summary=schedule_summary)
                    else:
                        return self.scheduling_agent.fu_schedule_suggestion.format(
                            schedule_summary=schedule_summary
                        )
                except:
                    return 'Your test schedules: ' + schedule_summary


    @staticmethod
    def _schedule_signature(schedule: dict) -> tuple:
        """
        Build a comparable key from the patient-visible parts of a schedule (test slots and follow-up slot).

        Args:
            schedule (dict): The scheduler output containing `test_schedule` and optionally `fu_schedule`.

        Returns:
            tuple: Order-independent signature of the test slots and the follow-up slot.
        """
        tests = tuple(sorted(
            (t['code'], t['date'], round(float(t['start']), 4), round(float(t['end']), 4))
            for t in schedule['test_schedule']
        ))
        fu_slot = schedule.get('fu_schedule')
        fu = tuple(sorted(
            (doctor, s['date'], round(float(s['start']), 4), round(float(s['end']), 4))
            for doctor, s in (fu_slot.items() if isinstance(fu_slot, dict) else [])
        ))
        return tests, fu


    @staticmethod
    def _copy_schedule_for_metrics(schedule: dict) -> dict:
        """
        Shallow-copy the scheduler output for NegotiationMetrics.

        Isolates the metrics' stored `_P` from the later in-place mutation of `pred_schedule`.

        Args:
            schedule (dict): The scheduler output (already list / float / sorted).

        Returns:
            dict: A shallow copy with a fresh `test_schedule` list of copied slot dicts.
        """
        return {**schedule, 'test_schedule': [{**t} for t in (schedule.get('test_schedule') or [])]}
    
    
    @staticmethod
    def postprocessing(strategy: str,
                       data: Union[str, dict],
                       filtered_doctor_information: Optional[dict] = None,
                       required_tests: Optional[list] = None,
                       filtered_test_device_information: Optional[dict] = None,
                       utc_offset: Optional[str] = None,
                       rule: Optional[SchedulingRule] = None,
                       attending_physician: Optional[str] = None) -> Optional[Union[str, dict]]:
        """
        Normalize a staff test-schedule output into the `pred_schedule` shape (start-sorted hour-float test slots).

        The `reasoning` branch computes `fu_schedule` deterministically, not from the LLM output.

        Args:
            strategy (str): `reasoning` (parse and enrich JSON text) or `tool_calling` (convert a tool result).
            data (Union[str, dict]): The output to post-process, possibly JSON text.
            filtered_doctor_information (Optional[dict], optional): Department-filtered doctor information. Defaults to None.
            required_tests (Optional[list], optional): Required-test metadata (reasoning only). Defaults to None.
            filtered_test_device_information (Optional[dict], optional): Maps device codes back to test codes (reasoning only). Defaults to None.
            utc_offset (Optional[str], optional): UTC offset for ISO timestamps (reasoning only). Defaults to None.
            rule (Optional[SchedulingRule], optional): Rule used to compute the follow-up slot (reasoning only). Defaults to None.
            attending_physician (Optional[str], optional): Physician for the follow-up slot (reasoning only). Defaults to None.

        Returns:
            Optional[Union[str, dict]]: The schedule dict, the input as a string if `reasoning` parsing fails,
                                        or None for an unknown strategy.
        """
        if strategy == 'reasoning':
            try:
                if isinstance(data, str):
                    match = re.search(r'```json\s*(\{.*?\})\s*```', data, re.DOTALL)
                    if match:
                        text_dict = json.loads(match.group(1))
                    else:
                        try:
                            text_dict = json.loads(data)
                        except:
                            return data
                else:
                    text_dict = data

                assert 'test_schedule' in text_dict
                assert isinstance(text_dict['test_schedule'], list)

                code_to_test = {t['code']: t for t in (required_tests or [])}
                device_to_code = {
                    dev: code
                    for code, info in (filtered_test_device_information or {}).get('test', {}).items()
                    for dev in info.get('devices', {})
                }

                latest, test_visit_dates, test_schedule = None, set(), []
                for entry in text_dict['test_schedule']:
                    assert isinstance(entry, dict) and len(entry) == 1
                    dev = next(iter(entry))
                    slot = entry[dev]
                    start = float(slot['start'])
                    end = float(slot['end'])
                    date = str(slot['date'])
                    code = device_to_code[dev]
                    t = code_to_test[code]
                    end_iso = get_iso_time(end, date, utc_offset)
                    result_ready_at = add_hours_to_iso(end_iso, t['result_hours'])
                    test_schedule.append({
                        'name': t['name'],
                        'code': code,
                        'device': dev,
                        'date': date,
                        'start': start,
                        'end': end,
                        'result_ready_at': result_ready_at,
                        'priority': t['priority'],
                    })
                    test_visit_dates.add(date)
                    if latest is None or compare_iso_time(result_ready_at, latest):
                        latest = result_ready_at
                
                # Allocate schedule information
                text_dict['preference_type'] = None
                text_dict['applied_unavailable'] = None   # the reasoning fallback reads the constraint from its prompt, not as a structured argument
                text_dict['test_schedule'] = test_schedule
                text_dict['test_visit_dates'] = list(test_visit_dates)
                text_dict['idle_waiting_time'] = calculate_idle_wait(test_schedule)
                text_dict['all_results_ready_at'] = latest

                # Follow-up consultation slot — computed deterministically here, NOT taken from the LLM.
                text_dict['fu_schedule'] = None
                if (
                    rule is not None
                    and attending_physician
                    and filtered_doctor_information
                    and latest is not None
                ):
                    candidates = rule.physician_filter(
                        filtered_doctor_information,
                        preferred_doctor=attending_physician,
                        min_time=latest,
                    )
                    earliest = rule.find_earliest_time(candidates)
                    if earliest['doctor'] and earliest['schedule']:
                        doctor = earliest['doctor'][0]
                        iso = earliest['schedule'][0]
                        duration = filtered_doctor_information['doctor'][doctor]['outpatient_duration']
                        st_hour = iso_to_hour(iso)
                        end_hour = float(Decimal(str(st_hour)) + Decimal(str(duration)))
                        text_dict['fu_schedule'] = {
                            doctor: {
                                'date': iso_to_date(iso),
                                'start': st_hour,
                                'end': end_hour,
                            }
                        }
                text_dict['test_schedule'].sort(key=lambda x: (x['date'], x['start']))
                return text_dict

            except Exception as e:
                log(colorstr('red', f'[postprocessing:reasoning] failed to parse/enrich schedule '
                                    f'({type(e).__name__}: {e}); returning raw output'), level='warning')
                return str(data)
        
        elif strategy == 'tool_calling':
            schedule = {
                'preference_type': data['preference_type'],
                'test_schedule': [],
                'test_visit_dates': data['test_visit_dates'],
                'idle_waiting_time': data['idle_waiting_time'],
                'fu_schedule': None,
                'all_results_ready_at': data['all_results_ready_at'],
                'applied_unavailable': data['applied_unavailable'],
            }
            
            # Follow-up visit schedule post-processing
            fu_schedule = data['fu_schedule']
            if fu_schedule['doctor'] and fu_schedule['schedule']:
                doctor = fu_schedule['doctor'][0]
                duration = filtered_doctor_information['doctor'][doctor]['outpatient_duration']
                date, st_hour = iso_to_date(fu_schedule['schedule'][0]), iso_to_hour(fu_schedule['schedule'][0])
                tr_hour = float(Decimal(str(duration)) + Decimal(str(st_hour)))
                schedule['fu_schedule'] = {doctor: {'date': date, 'start': st_hour, 'end': tr_hour}}
            
            # Test post-processing
            required_tests = data['test_schedule']
            for _, values in required_tests.items():
                st_hour, tr_hour = iso_to_hour(values['start']), iso_to_hour(values['end'])
                tmp_schedule = {**values}
                tmp_schedule['start'] = st_hour
                tmp_schedule['end'] = tr_hour
                schedule['test_schedule'].append(tmp_schedule)
            schedule['test_schedule'].sort(key=lambda x: (x['date'], x['start']))
            return schedule


    def _unavailable_prompt_field(self, known_condition: dict) -> str:
        """
        Fill the reasoning fallback's "patient unavailability" section.

        Uses the booking's stored constraint when present (rescheduling), else the dialogue to read it from.

        Args:
            known_condition (dict): Patient conditions known to the staff.

        Returns:
            str: Text for the prompt's `{UNAVAILABLE}` placeholder.
        """
        if 'unavailable' in known_condition:
            return describe_unavailable(known_condition['unavailable'], audience='staff')

        dialog = preprocess_dialog(self.dialog_history.get('test_scheduling') or [])
        if not dialog:
            return OPFU_UNAVAILABLE_PHRASE_STAFF['none']
        return ('Read it off the conversation below: honor every date — or morning / afternoon of a date — the '
                'patient said they cannot come in, and treat it as no constraint if they mentioned none.\n'
                f'```\n{dialog}\n```')


    def _format_test_devices_markdown(self, filtered_test_device_information: dict) -> str:
        """
        Render per-(test, device) markdown tables of free windows for the reasoning prompt.

        Dates blocked by the `restrict` policy are shown fully booked.

        Args:
            filtered_test_device_information (dict): `{'test': {code: {'duration_hour', 'devices': {device: {'schedule': {date: [...]}}}}}}`.

        Returns:
            str: Markdown tables of free windows, or a placeholder when there are no devices.
        """
        blocked = self.rules.blocked

        def bookable(code: str, schedule: dict) -> dict:
            return {date: [[self._START_HOUR, self._END_HOUR]] if (code, date) in blocked else slots
                    for date, slots in schedule.items()}

        items = [
            (f"Device `{device}` — supports test `{code}` (duration {info.get('duration_hour')}h)",
             bookable(code, dinfo.get('schedule') or {}))
            for code, info in (filtered_test_device_information.get('test') or {}).items()
            for device, dinfo in (info.get('devices') or {}).items()
        ]
        return render_availability_tables(items, self._START_HOUR, self._END_HOUR)


    def _apply_restrictions(self,
                            patient_name: str,
                            exclude: Optional[dict] = None) -> frozenset:
        """
        Hold the `restrict` policies' blocked (test, date) pairs for this patient on the rule (cleared without one).

        Args:
            patient_name (str): The patient's name, for `ctx.patient`.
            exclude (Optional[dict], optional): The patient's own booking being rescheduled. Defaults to None.

        Returns:
            frozenset: The blocked `(test_code, date)` pairs now held on the rule.
        """
        # No `restrict` policy: clear any previous restriction
        if not self.policies.restrictors:
            self._restrictions = {}
            return self.rules.set_blocked(())
        
        # Apply the union of every `restrict` policy
        record = self.patient_records[patient_name]
        context = RestrictContext(
            current_time=self.environment.current_time,
            start_hour=self._START_HOUR,
            utc_offset=self.rules._utc_offset,
            test_capacity=self.environment.test_capacity,
            test_duration=self.environment.test_duration,
            booked_hours=self.environment.booked_test_hours(exclude=exclude),
            patient=record,
        )
        self._restrictions = self.policies.restrict_by_policy(context)
        return self.rules.set_blocked(frozenset().union(*self._restrictions.values()))


    def _restriction_record(self, test_codes: list) -> Optional[dict]:
        """
        Per-booking record of what each `restrict` policy blocked for this booking's tests (stored as `restrictions`).

        Args:
            test_codes (list): The booking's test codes.

        Returns:
            Optional[dict]: ``{policy id: [[test_code, date], ...]}`` (sorted; empty lists for policies that blocked none
                            of these tests), or None without a `restrict` policy.
        """
        if not self.policies.restrictors:
            return None
        codes = set(test_codes)
        return {pid: sorted([c, d] for c, d in pairs if c in codes) for pid, pairs in self._restrictions.items()}


    def _capacity_protection_record(self,
                                    preference_type: str,
                                    test_codes: list,
                                    test_device_information: Optional[dict] = None,
                                    unavailable: Optional[dict] = None) -> Optional[dict]:
        """
        Per-booking `capacity_protection` record of the `restrict` policy.

        When pairs were blocked, `unprotected` is the schedule the same objective gives with nothing blocked.

        Args:
            preference_type (str): Objective the booked schedule was built with (`throughput_max` after a negotiation).
            test_codes (list): The patient's required test codes.
            test_device_information (Optional[dict], optional): Test device schedules (None under FHIR). Defaults to None.
            unavailable (Optional[dict], optional): The patient's test-time unavailability. Defaults to None.

        Returns:
            Optional[dict]: `{'blocked': [[code, date], ...], 'objective': preference_type, 'unprotected': {...} | None}`,
                            or None without a `restrict` policy. `unprotected.tests` keeps each placed test's date and result
                            time, so evaluation can compare it with a partial booking on the same tests.
        """
        if not self.policies.restrictors:
            return None

        blocked = sorted([code, date] for code, date in self.rules.blocked if code in test_codes)
        record = {'blocked': blocked, 'objective': preference_type, 'unprotected': None}
        if blocked:
            filtered_test_device_information = self.environment.get_test_device_schedule(
                test_device_information=test_device_information,
                test_code=list(test_codes),
                fhir_integration=self.fhir_integration and test_device_information is None,
            )
            with self.rules.restrictions_lifted():
                unprotected = self.rules.schedule_tests(
                    preference_type, filtered_test_device_information, list(test_codes), 10, unavailable=unavailable,
                )
            record['unprotected'] = {
                **{k: unprotected[k] for k in ('all_results_ready_at', 'test_visit_dates', 'idle_waiting_time', 'unscheduled')},
                'tests': {code: {'date': a['date'], 'result_ready_at': a['result_ready_at']}
                          for code, a in unprotected['test_schedule'].items()},
            }
        return record


    def _get_rescheduled_test_result(self,
                                     known_condition: dict,
                                     doctor_information: Optional[dict] = None,
                                     test_device_information: Optional[dict] = None,
                                     **kwargs) -> dict:
        """
        Re-run whole-test-set scheduling for a booking to find an improved schedule.

        Args:
            known_condition (dict): The original follow-up booking being rescheduled.
            doctor_information (Optional[dict], optional): Doctor information (required without FHIR). Defaults to None.
            test_device_information (Optional[dict], optional): Test device schedules. Defaults to None.
            **kwargs: Additional keyword arguments forwarded to `test_scheduling`.

        Returns:
            dict: Rescheduled test-set schedule (test-scheduling shape).
        """
        # Sanity check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        required_test_codes = [t['code'] for t in known_condition.get('test') or []]

        # Restrictions for the booking's patient; its current hours are about to be freed
        self._apply_restrictions(known_condition['patient'], exclude=known_condition)
        filtered_doctor_information = self.environment.get_doctor_schedule(
            doctor_information=doctor_information,
            department=known_condition['department'],
            fhir_integration=self.fhir_integration and doctor_information is None,
        )
        filtered_test_device_information = self.environment.get_test_device_schedule(
            test_device_information=test_device_information,
            test_code=required_test_codes,
            fhir_integration=self.fhir_integration and test_device_information is None,
        )

        # The agent picks the follow-up test-scheduling tool
        _schedule_client = self.scheduling_agent.build_agent(
            rule=self.rules,
            doctor_info=filtered_doctor_information,
            only_schedule_tool=True,
            required_test_codes=required_test_codes,
            test_device_information=filtered_test_device_information,
            patient_unavailable=known_condition.get('unavailable'),
        )
        # Express the original preference and re-supply required tests for the reasoning fallback
        known_condition = {
            **known_condition,
            'required_tests': [{'test_code': t['code']} for t in known_condition.get('test') or []],
        }
        new_schedule = self.test_scheduling(
            client=_schedule_client,
            known_condition=known_condition,
            doctor_information=doctor_information,
            test_device_information=test_device_information,
            reschedule_flag=True,
            **kwargs,
        )['result']
        return new_schedule


    def _check_test_reschedule_validity(self,
                                        idx: int,
                                        new_schedule: dict,
                                        original_schedule: dict,
                                        doctor_information: dict,
                                        test_device_information: dict) -> Optional[dict]:
        """
        Check the rescheduling availability under the booking's preference-based improvement criterion.

        Args:
            idx (int): Index of the requested booking (original booking index).
            new_schedule (dict): New candidate test-set schedule.
            original_schedule (dict): The original follow-up booking.
            doctor_information (dict): A dictionary containing information about the doctor(s).
            test_device_information (dict): Test device information containing device schedules.

        Returns:
            Optional[dict]: New follow-up prediction if the rescheduling improves the booking; otherwise None.
        """
        # A partial re-schedule that drops any test is never an improvement
        original_tests = original_schedule.get('test') or []
        if {t['code'] for t in new_schedule['test_schedule']} != {t['code'] for t in original_tests}:
            return None

        # Original metrics
        original_dates = {t['date'] for t in original_tests}
        original_ready = None
        for t in original_tests:
            if t.get('result_ready_at') and (original_ready is None or compare_iso_time(t['result_ready_at'], original_ready)):
                original_ready = t['result_ready_at']
        new_ready = new_schedule.get('all_results_ready_at')
        new_dates = set(new_schedule.get('test_visit_dates') or [])

        # `new_t` is strictly earlier than `old_t`
        def _strictly_earlier(new_t: Optional[str] = None, old_t: Optional[str] = None) -> bool:
            return new_t is not None and old_t is not None and compare_iso_time(old_t, new_t) and new_t != old_t

        # Preference-based improvement
        preference = original_schedule.get('preference')
        if preference == 'indifferent':
            preference = 'throughput_max'
        
        if preference == 'visit_min':
            improved = len(new_dates) < len(original_dates) or \
                (len(new_dates) == len(original_dates) and _strictly_earlier(new_ready, original_ready))
        elif preference == 'stay_min':
            new_idle = new_schedule['idle_waiting_time']
            original_idle = original_schedule['idle_waiting_time']
            improved = new_idle < original_idle
        else:
            improved = _strictly_earlier(new_ready, original_ready)

        if not improved:
            return None

        # Restriction record of the new booking, taken before the old booking's slots are freed (as in its search)
        protection_record = self._capacity_protection_record(
            new_schedule.get('preference_type') or preference,
            [t['code'] for t in original_tests], test_device_information, original_schedule.get('unavailable'),
        )

        # Free the old booking, then build the new follow-up prediction (booking is done by the caller)
        self.rules.cancel_test_schedule(idx, doctor_information, test_device_information, original_schedule)
        fu_slot = new_schedule['fu_schedule']
        fu_schedule = fu_slot[next(iter(fu_slot))] if fu_slot else None
        for item in new_schedule['test_schedule']:
            item['schedule'] = [item.pop('start'), item.pop('end')]

        final_schedule = {
            'visit_type': 'follow_up_visit',
            'patient': original_schedule['patient'],
            'attending_physician': original_schedule['attending_physician'],
            'department': original_schedule['department'],
            'date': fu_schedule['date'] if fu_schedule else None,
            'schedule': [fu_schedule['start'], fu_schedule['end']] if fu_schedule else None,
            'patient_intention': original_schedule.get('patient_intention'),
            'preference': original_schedule.get('preference'),
            'urgency': original_schedule.get('urgency'),
            'preferred_doctor': original_schedule.get('preferred_doctor'),
            'valid_from': original_schedule.get('valid_from'),
            'unavailable': original_schedule.get('unavailable'),
            'test': new_schedule['test_schedule'],
            'idle_waiting_time': new_schedule['idle_waiting_time'],
            'capacity_protection': protection_record,
            'restrictions': self._restriction_record([t['code'] for t in original_tests]),
            'last_updated_time': self.environment.current_time,
        }
        return final_schedule


    def _make_test_reschedule_pipeline(self,
                                       doctor_information: Optional[dict] = None,
                                       test_device_information: Optional[dict] = None,
                                       **kwargs) -> Callable[[int, dict], dict]:
        """
        Build the post-retrieval test rescheduling pipeline (reschedule -> sanity check -> validity -> waiting list).

        Args:
            doctor_information (Optional[dict], optional): Doctor information. Defaults to None.
            test_device_information (Optional[dict], optional): Test device schedules. Defaults to None.
            **kwargs: Additional keyword arguments forwarded to the inner scheduling agent.

        Returns:
            Callable[[int, dict], dict]: ``(idx, original_schedule) -> {'action', 'new_schedule', 'status_code'}``,
                                         action one of 'reschedule' / 'waiting_list' / 'schedule_fail'.
        """
        def pipeline(idx: int, original_schedule: dict) -> dict:
            try:
                new_schedule = self._get_rescheduled_test_result(
                    known_condition=original_schedule,
                    doctor_information=doctor_information,
                    test_device_information=test_device_information,
                    **kwargs,
                )
            except Exception:
                return {'action': 'schedule_fail', 'new_schedule': None,
                        'status_code': STATUS_CODES['format']}

            if self.sanity_checker is not None:
                filtered_test_device_information = self.environment.get_test_device_schedule(
                    test_device_information=test_device_information,
                    test_code=[t['code'] for t in original_schedule.get('test') or []],
                    fhir_integration=self.fhir_integration and test_device_information is None,
                )
                ok, code = self.sanity_checker.test_schedule_check(
                    prediction=new_schedule,
                    gt_patient_condition=original_schedule,
                    test_device_information=filtered_test_device_information,
                    doctor_information=doctor_information,
                    environment=self.environment,
                    rule=self.rules,
                )
                if not ok:
                    return {'action': 'schedule_fail', 'new_schedule': new_schedule,
                            'status_code': code}

            try:
                final = self._check_test_reschedule_validity(
                    idx=idx,
                    new_schedule=new_schedule,
                    original_schedule=original_schedule,
                    doctor_information=doctor_information,
                    test_device_information=test_device_information,
                )
                if final is not None:
                    return {'action': 'reschedule', 'new_schedule': final, 'status_code': None}
                self.environment.add_waiting_list(idx, True)
                return {'action': 'waiting_list', 'new_schedule': None, 'status_code': None}
            except Exception:
                return {'action': 'schedule_fail', 'new_schedule': new_schedule,
                        'status_code': STATUS_CODES['format']}

        return pipeline


    def automatic_waiting_list_update(self,
                                      doctor_information: dict,
                                      test_device_information: dict,
                                      **kwargs) -> Iterator[dict]:
        """
        Try to improve each follow-up waiting-list booking; a failure leaves the entry on the list.

        Args:
            doctor_information (dict): Doctor information.
            test_device_information (dict): Test device schedules.
            **kwargs: Additional keyword arguments forwarded to `test_scheduling`.

        Yields:
            dict: ``{'doctor_information', 'test_device_information', 'result_dict', 'original'}`` per attempted entry.
        """
        # Snapshot the list: a successful reschedule pops the entry from `waiting_list` mid-iteration
        for turn, (idx, original) in enumerate(list(self.environment.waiting_list)):
            if original['status'] == SCHEDULE_STATUS['scheduled'] and original.get('visit_type') == 'follow_up_visit':
                # A tool-calling / reasoning-fallback failure only means this waiting entry could not be
                # improved this round; it stays on the waiting list and must not end the whole simulation.
                try:
                    new_schedule = self._get_rescheduled_test_result(
                        known_condition=original,
                        doctor_information=doctor_information,
                        test_device_information=test_device_information,
                        **kwargs
                    )
                except Exception as e:
                    log(f'Automatic waiting list update failed to produce a schedule: {e}', level='warning')
                    result_dict = {
                        'gt': ['automatic rescheduling'],
                        'pred': [None],
                        'status': [False],
                        'status_code': [STATUS_CODES['reschedule']['schedule'].format(status_code=STATUS_CODES['format'])],
                        'dialog': ['automatic waiting list update from the system']
                    }
                    yield {'doctor_information': doctor_information, 'test_device_information': test_device_information, 'result_dict': result_dict, 'original': original}
                    continue

                # Sanity check
                ## No GT case
                if self.sanity_checker is None:
                    status, status_code = True, STATUS_CODES['correct']
                else:
                    filtered_test_device_information = self.environment.get_test_device_schedule(
                        test_device_information=test_device_information,
                        test_code=[t['code'] for t in original.get('test') or []],
                        fhir_integration=self.fhir_integration and test_device_information is None,
                    )
                    status, status_code = self.sanity_checker.test_schedule_check(
                        prediction=new_schedule,
                        gt_patient_condition=original,
                        test_device_information=filtered_test_device_information,
                        doctor_information=doctor_information,
                        environment=self.environment,
                        rule=self.rules,
                    )

                if status:
                    try:
                        final_schedule = self._check_test_reschedule_validity(
                            idx=idx,
                            new_schedule=new_schedule,
                            original_schedule=original,
                            doctor_information=doctor_information,
                            test_device_information=test_device_information,
                        )
                        if final_schedule is not None:
                            result_dict = {
                                'gt': ['automatic rescheduling'],
                                'pred': [final_schedule],
                                'status': [True],
                                'status_code': [STATUS_CODES['correct']],
                                'dialog': ['automatic waiting list update from the system']
                            }
                            yield {'doctor_information': doctor_information, 'test_device_information': test_device_information, 'result_dict': result_dict, 'original': original}

                    except:
                        log('No sanity checker is available; an error occurred while parsing the prediction. Returning a failure result.', level='warning')
                        result_dict = {
                            'gt': ['automatic rescheduling'],
                            'pred': [new_schedule],
                            'status': [False],
                            'status_code': [STATUS_CODES['reschedule']['schedule'].format(status_code=STATUS_CODES['format'])],
                            'dialog': ['automatic waiting list update from the system']
                        }
                        yield {'doctor_information': doctor_information, 'test_device_information': test_device_information, 'result_dict': result_dict, 'original': original}

                else:
                    result_dict = {
                        'gt': ['automatic rescheduling'],
                        'pred': [new_schedule],
                        'status': [status],
                        'status_code': [STATUS_CODES['reschedule']['schedule'].format(status_code=status_code)],
                        'dialog': ['automatic waiting list update from the system']
                    }
                    yield {'doctor_information': doctor_information, 'test_device_information': test_device_information, 'result_dict': result_dict, 'original': original}


    def _calculate_negotiation_metrics(self,
                                       patient_preferred_schedule: dict,
                                       filtered_test_device_information: dict,
                                       patient_name: str,
                                       preference: Optional[str] = None,
                                       unavailable: Optional[dict] = None) -> dict:
        """
        Compute the negotiation metrics of the preferred schedule vs. the throughput_max counterfactual and record
        the `review` policy's decision (`negotiation_action` / `do_negotiate` / `decided_by`, plus its own fields).

        Args:
            patient_preferred_schedule (dict): The preference-based test schedule the staff proposed.
            filtered_test_device_information (dict): Device schedules for the required tests.
            patient_name (str): The patient's name, to attach their record as `ctx.patient`.
            preference (Optional[str], optional): Preference to score against; None uses the schedule's
                                                  `preference_type`. Defaults to None.
            unavailable (Optional[dict], optional): The patient's test-time unavailability, applied to the
                                                    throughput_max counterfactual. Defaults to None.

        Returns:
            dict: The negotiation metrics (`preference`, `gpc`, `tcl`, `ti`, `do_negotiate`, ...).
        """
        preference = preference or patient_preferred_schedule.get('preference_type')

        # The hospital-fixed constants (e.g. tcl_temperature) come from the `review` policy.
        metrics = NegotiationMetrics(
            preference=preference,
            achieved_schedule=self._copy_schedule_for_metrics(patient_preferred_schedule),
            filtered_test_device_information=filtered_test_device_information,
            rule=self.rules,
            environment=self.environment,
            unavailable=unavailable,
            **self.policies.metrics_params(preference),
        ).to_dict()

        # User-defined metrics are computed from the built-in ones, then shown to the `review` policies with them
        record = self.patient_records[patient_name]
        context = ReviewContext(
            preference=preference,
            metrics=metrics,
            patient=record,
            schedules={'preferred': metrics.get('_P'), 'hospital': metrics.get('_T')},
        )
        custom = self.policies.compute_metrics(context)
        metrics.update(custom)

        # The `review` policy makes the final call: 'keep' (book preferred), 'auto' (book its objective directly), or 'negotiate' (persuade).
        decision = self.policies.review(replace(context, metrics=metrics) if custom else context)
        self._last_decision = decision
        metrics.update({
            'ti': None,
            **decision.record,  # τ policy's record overrides it with its TI in threshold mode
            'negotiation_action': decision.action,
            'do_negotiate': decision.do_negotiate,
            'decided_by': decision.decided_by,
        })
        _metrics = {k: v for k, v in metrics.items() if not k.startswith('_')}
        log(colorstr('cyan', f'[negotiation_metrics] {_metrics}'))
        return metrics
    

    @staticmethod
    def _wrap_negotiation_turn(template: str, negotiation_round: int, utterance: str) -> str:
        """
        Fill a per-round negotiation wrapper (`STAFF_NEGOTIATION_TURN` / `PATIENT_NEGOTIATION_TURN`).

        Uses `str.replace`, so braces in `utterance` are safe; `{round}` is substituted first.

        Args:
            template (str): Wrapper with `{round}` and `{utterance}` placeholders.
            negotiation_round (int): 1-based negotiation round.
            utterance (str): The utterance to wrap.

        Returns:
            str: The filled wrapper.
        """
        return template.replace('{round}', str(negotiation_round)).replace('{utterance}', utterance)


    def _negotiating(self,
                     patient_preference: str,
                     negotiation_metrics: dict,
                     negotiation_round: int) -> Information:
        """
        Run one negotiation round: the staff persuasion agent responds to the patient's latest utterance.

        The agent ends its message with `#ACCEPT` once the patient agrees (or at the forced-close round).

        Args:
            patient_preference (str): The patient's stated preference for scheduling (e.g., 'throughput_max', 'visit_min', 'stay_min').
            negotiation_metrics (dict): Metrics computed by `_calculate_negotiation_metrics`, including `gpc`, `tcl`, `ti`, and other relevant values.
            negotiation_round (int): 1-based index of the persuasion round about to be produced.

        Returns:
            Information: The staff utterance for this negotiation round.
        """
        # Negotiation algorithm
        negotiation_agent = self.init_negotiation_agent(
            patient_preference=patient_preference,
            negotiation_metrics=negotiation_metrics,
        )

        # Respond to the patient's latest utterance (persuade, or accept with an `#ACCEPT` tag), giving the
        # staff the current round so it can force-close after enough continued refusal.
        user_prompt = self._wrap_negotiation_turn(
            self.STAFF_NEGOTIATION_TURN,
            negotiation_round,
            self.dialog_history['test_scheduling'][-1]['content'],
        )
        _pitch = negotiation_agent(
            user_prompt=user_prompt,
            using_multi_turn=True,
            verbose=False,
        )
        pitch = Information(
            response=_pitch,
            agent=negotiation_agent.target_task,
        )
        return pitch


    def _throughput_pred_from_metrics(self,
                                      negotiation_metrics: dict,
                                      doctor_information: Optional[dict],
                                      department: str,
                                      attending_physician: str) -> dict:
        """
        Build a bookable throughput_max schedule from the already-computed counterfactual `_T`.

        Test slots are copied (the stored `_T` is not mutated) and the follow-up slot is added as in `postprocessing`.

        Args:
            negotiation_metrics (dict): Metrics carrying `_T`, the throughput_max counterfactual schedule.
            doctor_information (Optional[dict]): Doctor schedules (None under FHIR).
            department (str): The patient's department.
            attending_physician (str): The attending physician who ordered the tests.

        Returns:
            dict: A `pred_schedule`-shaped throughput_max schedule (with `fu_schedule`).
        """
        t = negotiation_metrics['_T']
        pred = {
            'preference_type': 'throughput_max',
            'test_schedule': [{**x} for x in (t.get('test_schedule') or [])],
            'test_visit_dates': list(t.get('test_visit_dates') or []),
            'idle_waiting_time': t.get('idle_waiting_time'),
            'all_results_ready_at': t.get('all_results_ready_at'),
            'applied_unavailable': t.get('applied_unavailable'),
            'fu_schedule': None,
        }
        latest = pred['all_results_ready_at']
        if latest and attending_physician:
            filtered_doctor_information = self.environment.get_doctor_schedule(
                doctor_information=doctor_information,
                department=department,
                fhir_integration=self.fhir_integration and doctor_information is None,
            )
            candidates = self.rules.physician_filter(
                filtered_doctor_information, preferred_doctor=attending_physician, min_time=latest,
            )
            earliest = self.rules.find_earliest_time(candidates)
            if earliest['doctor'] and earliest['schedule']:
                doctor = earliest['doctor'][0]
                iso = earliest['schedule'][0]
                duration = filtered_doctor_information['doctor'][doctor]['outpatient_duration']
                st_hour = iso_to_hour(iso)
                end_hour = float(Decimal(str(st_hour)) + Decimal(str(duration)))
                pred['fu_schedule'] = {doctor: {'date': iso_to_date(iso), 'start': st_hour, 'end': end_hour}}
        return pred


    def test_scheduling(self,
                        client: AgentExecutor,
                        known_condition: dict,
                        doctor_information: Optional[dict] = None, 
                        test_device_information: Optional[dict] = None, 
                        reschedule_flag: bool = False,
                        chat_history: list = [],
                        reasoning_max_tries: int = 0,
                        **kwargs) -> dict:
        """
        Make a test appointment between the doctor and the patient.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            known_condition (dict): Patient conditions known to the staff.
            doctor_information (Optional[dict], optional): Doctor information (required without FHIR). Defaults to None.
            test_device_information (Optional[dict], optional): Test device schedules (required without FHIR). Defaults to None.
            reschedule_flag (bool, optional): Whether this process is rescheduling or not. Defaults to False.
            chat_history (list, optional): Chat history. Defaults to [].
            reasoning_max_tries (int, optional): Extra reasoning-fallback attempts after an unparsable output. Defaults to 0.
            **kwargs: Forwarded to the reasoning-fallback agent; ``callback`` is popped for tool calling.

        Returns:
            dict: Scheduling result (``type`` 'tool' with a schedule/retrieval result, or 'text' clarification).

        Raises:
            ToolCallingError: If no appropriate tool was chosen, or the reasoning fallback has no retrieved tests.
            TypeError: On an unexpected tool-calling result type (caught here; falls back to reasoning).
            SchedulingError: If the reasoning fallback produces no valid schedule JSON.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")
            assert test_device_information is not None, colorstr("red", f"Test device information must be provided if you don't use FHIR.")

        # Initialization based on the known condition from the staff
        callback = kwargs.pop('callback', None)
        department = known_condition['department']
        attending_physician = known_condition['attending_physician']
        
        # First, try to use the tool calling
        try:
            assert self.scheduling_strategy == 'tool_calling', log('Scheduling strategy is set to `reasoning`, directly use the reasoning method.', level='warning')
            
            # Invoke
            prediction = scheduling_tool_calling(
                client=client, 
                user_prompt = known_condition['patient_intention'] + f' (Attending Physician: {attending_physician})' \
                    if attending_physician else known_condition['patient_intention'],
                history=chat_history,
                callback=callback,
            )

            # Scheduling result
            if prediction['type'] == 'tool':
                res = prediction['result']
                if res['action'] == 'retrieval':
                    # Patient information not found case: -> text
                    if res['status'] is None and res['index']['pred'] == -1:
                        prediction['type'] = 'text'
                        prediction['result'] = "Sorry, we couldn't find a matching information. Could you please check your details again (patient and doctor names)?"
                        return prediction

                    # Identification failure ends the run; the loop surfaces it as a raise.
                    # Recording the outcome is left to `test_scheduling_simulate`.
                    if res['status'] is False:
                        prediction['tmp_flag'] = 'retrieve'
                    return prediction

                elif res['action'] == 'clarification':
                    prediction['type'] = 'text'
                    prediction['result'] = res['message']
                    return prediction

                elif res['action'] == 'scheduling':
                    filtered_doctor_information = self.environment.get_doctor_schedule(
                        doctor_information=doctor_information,
                        department=department,
                        fhir_integration=self.fhir_integration and doctor_information is None,
                    )
                    test_schedules = OPFUSchedulingSimulation.postprocessing(
                        strategy='tool_calling',
                        data=res,
                        filtered_doctor_information=filtered_doctor_information,
                    )
                    prediction['result'] = test_schedules

                else:
                    raise ToolCallingError(colorstr('red', 'Failed to choose an appropriate scheduling tool.'))
            
            ## Dialogue
            elif prediction['type'] == 'text':
                if 'no tool' in prediction['result'].lower():
                    raise ToolCallingError(colorstr('red', 'Failed to choose an appropriate scheduling tool.'))
            
            ## Error
            else:
                raise TypeError(colorstr("red", "Error: Unexpected return type from scheduling method."))

        # If tool calling fails, fallback to LLM-based scheduling
        except Exception as e:
            log(f'Exception occured: {e}', 'warning')
            # Fallback only applies after retrieve_tests_after_first_visit has populated required_tests.
            if not known_condition.get('test'):
                raise ToolCallingError(colorstr('red', 'Reasoning fallback without test retrieval.'))

            if self.scheduling_strategy == 'tool_calling':
                log('Failed to select an appropriate tool. Falling back to reasoning-based scheduling.', level='warning')

            required_test_codes = [t['code'] for t in known_condition['test']]
            filtered_doctor_information = self.environment.get_doctor_schedule(
                doctor_information=doctor_information,
                department=department,
                fhir_integration=self.fhir_integration and doctor_information is None,
            )
            filtered_test_device_information = self.environment.get_test_device_schedule(
                test_device_information=test_device_information,
                test_code=required_test_codes,
                fhir_integration=self.fhir_integration and test_device_information is None,
            )
            current_time = f"{self.environment.current_time} (Date: {iso_to_date(self.environment.current_time)}, Time: {round(iso_to_hour(self.environment.current_time), 3)})"
            user_prompt = self.scheduling_agent.scheduling_user_prompt_template.format(
                START_HOUR=self._START_HOUR,
                END_HOUR=self._END_HOUR,
                TIME_UNIT=self._TIME_UNIT,
                CURRENT_TIME=current_time,
                DEPARTMENT=department,
                PREFERENCE=known_condition['patient_intention'],
                UNAVAILABLE=self._unavailable_prompt_field(known_condition),
                DAY=self._DAY,
                TESTS=json.dumps(known_condition['test'], indent=2),
                TEST_DEVICES=self._format_test_devices_markdown(filtered_test_device_information),
            )

            tries, schedule = 0, None
            while 1:
                schedule = self.scheduling_agent(
                    user_prompt,
                    using_multi_turn=False,
                    verbose=False,
                    **kwargs,
                )
                schedule = OPFUSchedulingSimulation.postprocessing(
                    strategy='reasoning',
                    data=schedule,
                    filtered_doctor_information=filtered_doctor_information,
                    required_tests=known_condition['test'],
                    filtered_test_device_information=filtered_test_device_information,
                    utc_offset=self.environment._utc_offset,
                    rule=self.rules,
                    attending_physician=known_condition.get('attending_physician'),
                )
                if isinstance(schedule, dict) or tries >= reasoning_max_tries:
                    break
                tries += 1

            if not isinstance(schedule, dict):
                self.scheduling_agent.reset_history(verbose=False)
                raise SchedulingError(colorstr('red', 'Reasoning fallback failed to produce a valid schedule JSON.'))

            prediction = {
                'type': 'tool',
                'result': schedule,
                'raw': None,
                'token': deepcopy(self.scheduling_agent.client.token_usages),
            }
            self.scheduling_agent.reset_history(verbose=False)

        return prediction


    def test_scheduling_simulate(self,
                                 gt_data: list[dict],
                                 staff_known_data: dict,
                                 doctor_information: Optional[dict] = None,
                                 test_device_information: Optional[dict] = None,
                                 verbose: bool = False,
                                 max_inferences: int = 15,
                                 natural_express: bool = True,
                                 reasoning_max_tries: int = 0,
                                 patient_kwargs: dict = {},
                                 staff_kwargs: dict = {},
                                 **kwargs) -> tuple[Optional[dict], Optional[dict], dict, dict]:
        """
        Simulate a multi-turn follow-up test-scheduling dialogue (with optional negotiation) between the patient and staff.

        Args:
            gt_data (list[dict]): Ground-truth patient conditions, one per preference (later ones follow rejections).
            staff_known_data (dict): Patient information known to the staff agent.
            doctor_information (Optional[dict], optional): Doctor information (required without FHIR). Defaults to None.
            test_device_information (Optional[dict], optional): Test device schedules (required without FHIR). Defaults to None.
            verbose (bool, optional): Whether to log detailed simulation outputs. Defaults to False.
            max_inferences (int, optional): Maximum number of dialogue turns. Defaults to 15.
            natural_express (bool, optional): Whether to express the new schedule naturally. Defaults to True.
            reasoning_max_tries (int, optional): Reasoning fallback maximum number of retries. Defaults to 0.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent. Defaults to {}.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff scheduling function. Defaults to {}.
            **kwargs: Shared keyword arguments passed to both agents.

        Returns:
            tuple[Optional[dict], Optional[dict], dict, dict]: Doctor information, test device information,
                                                               the result dictionary, and token statistics.

        Raises:
            TurnLimitReached: When the dialogue runs out of turns (caught here and recorded in the result).
            DataNotFoundError: When patient information is not found (caught here and recorded in the result).
            ToolSelectionError: When the staff picks a tool that ignores the preference (caught here and recorded).
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")
            assert test_device_information is not None, colorstr("red", f"Test device information must be provided if you don't use FHIR.")

        # Initialize agents and result dictionary
        result_dict = init_result_dict()
        self._init_agents(verbose=verbose)
        self._init_history()
        patient_token_stats, staff_token_stats = {}, {}
        staff_token_callback = TokenUsageCallback()
        patient_info = self.environment.patient_schedules
        filtered_doctor_information = None
        filtered_test_device_information = None
        required_test_codes = None
        _prev_suggested_schedule = None
        tool_calling_agent = self.scheduling_agent.build_agent(
            rule=self.rules, 
            doctor_info=None,
            patient_schedule_list=patient_info,
            gt_idx=gt_data[0]['index'],
        )
        merged_patient_kwargs = {**patient_kwargs, **kwargs}
        merged_staff_kwargs = {**staff_kwargs, **kwargs}

        # Restriction policy for this patient (hospital time is fixed until the booking)
        self._apply_restrictions(gt_data[0]['patient_fv']['patient'])

        # Staff turn closure: captures all of `scheduling`'s simulation-side arguments
        def staff_turn(user_prompt: str) -> tuple[str, dict]:
            staff_known_data.update({'patient_intention': user_prompt})
            prediction = self.test_scheduling(
                tool_calling_agent,
                staff_known_data,
                doctor_information,
                test_device_information,
                chat_history=self._to_lc_history('test_scheduling'),
                reasoning_max_tries=reasoning_max_tries,
                callback=staff_token_callback,
                **merged_staff_kwargs
            )
            reply = self._render_staff_reply(
                prediction, 'test_scheduling', gt_patient_condition, 
                staff_known_data, natural_express, _prev_suggested_schedule
            )
            return reply, prediction

        # Start conversation
        self._open_staff_turn('test_scheduling')

        # Iterate over multiple preferences if exists
        tries = 0
        preference_reject_prob = 0.0 if len(gt_data) <= 1 else self.preference_rejection_prob
        booked_preference = gt_data[0]['preference']
        try:
            # Preference iteration
            for i, gt_patient_condition in enumerate(gt_data):
                negotiation_ready = bool(staff_known_data.get('test'))  # Because of preference rejection logic
                negotiating_active = False
                negotiation_done = False
                negotiation_round = 0
                negotiation_outcome = 'none'   # 'accepted' | 'forced' | 'kept' | 'auto' | 'none'
                negotiation_metrics = self._init_negotiation_metrics(gt_patient_condition.get('preference'))

                # For the rejection scenario: the patient rejects what the staff actually arranged last time
                if i != 0:
                    self._update_patient_system_prompt(
                        patient_condition=gt_patient_condition,
                        rejected_preference=booked_preference
                    )

                while 1:
                    # Active negotiation
                    if negotiating_active:
                        # Patient negotiation response (the round number lets the patient escalate its irritation)
                        neg_prompt = self._wrap_negotiation_turn(
                            self.PATIENT_NEGOTIATION_TURN,
                            negotiation_round,
                            self.dialog_history['test_scheduling'][-1]['content'],
                        )
                        patient_response = self._patient_turn(
                            'test_scheduling',
                            f"{gt_patient_condition['preference']},{gt_patient_condition['distance']}",
                            prompt=neg_prompt,
                            **merged_patient_kwargs,
                        )
                        patient_token_stats = self.patient_agent.client.token_usages

                        # Staff negotiation response
                        pitch = self._negotiating(
                            patient_preference=gt_patient_condition['preference'],
                            negotiation_metrics=negotiation_metrics,
                            negotiation_round=negotiation_round + 1,
                        )
                        negotiation_round += 1   # this pitch is a round

                        if '#KEEP_PREFERENCE' in pitch.response:
                            # Staff gave up: the patient holds their preference.
                            negotiating_active = False
                            negotiation_outcome = 'kept'
                            pred_schedule = prediction['result']
                            pitch.response = self.NEGOTIATION_CLOSING_LEAD[negotiation_outcome] + ' ' + self._render_staff_reply(
                                {'type': 'tool', 'result': pred_schedule},
                                'test_scheduling', gt_patient_condition, staff_known_data, natural_express, _prev_suggested_schedule,
                            )
                            self._record_staff_turn('test_scheduling', pitch)
                            break

                        if '#FORCE_ACCEPT' in pitch.response or '#ACCEPT' in pitch.response:
                            # Negotiation closed: patient conceded ('accepted') or staff imposed it ('forced').
                            negotiating_active, negotiation_done = False, True
                            negotiation_outcome = 'forced' if '#FORCE_ACCEPT' in pitch.response else 'accepted'
                            pred_schedule = self._throughput_pred_from_metrics(
                                negotiation_metrics, doctor_information,
                                staff_known_data['department'], staff_known_data['attending_physician'],
                            )
                            pitch.response = self.NEGOTIATION_CLOSING_LEAD[negotiation_outcome] + ' ' + self._render_staff_reply(
                                {'type': 'tool', 'result': pred_schedule},
                                'test_scheduling', gt_patient_condition, staff_known_data, natural_express, _prev_suggested_schedule,
                            )
                            self._record_staff_turn('test_scheduling', pitch)
                            break
                        else:
                            # Still resisting: record the pitch and let the patient respond next round.
                            self._record_staff_turn('test_scheduling', pitch)
                            tries += 1
                            if tries > max_inferences:
                                raise TurnLimitReached
                            continue

                    # Regular turn: the patient speaks, then the staff schedules
                    else:
                        patient_response = self._patient_turn(
                            'test_scheduling',
                            f"{gt_patient_condition['preference']},{gt_patient_condition['distance']}",
                            **merged_patient_kwargs,
                        )
                        patient_token_stats = self.patient_agent.client.token_usages

                        # Scheduling from staff
                        output, prediction = self._staff_turn(patient_response, staff_turn)

                        # Token accounting
                        staff_token_stats = self._accumulate_staff_tokens(
                            prediction, staff_token_stats, staff_token_callback
                        )

                    # Start negotiating (once per preference): staff proposed the patient's preferred schedule -> enter the persuasion sub-loop.
                    if (
                        negotiation_ready
                        and not negotiation_done
                        and prediction['type'] == 'tool'
                        and isinstance(prediction['result'], dict)
                        and prediction['result'].get('preference_type') not in {None, 'throughput_max'}
                        and prediction['result'].get('preference_type') == gt_patient_condition['preference']
                    ):
                        negotiation_metrics = self._calculate_negotiation_metrics(
                            patient_preferred_schedule=prediction['result'],
                            filtered_test_device_information=filtered_test_device_information,
                            unavailable=gt_patient_condition['unavailable'],
                            patient_name=gt_data[0]['patient_fv']['patient'],
                        )
                        action = negotiation_metrics['negotiation_action']

                        if action == 'negotiate':
                            negotiating_active = True
                            pitch = self._negotiating(
                                patient_preference=gt_patient_condition['preference'],
                                negotiation_metrics=negotiation_metrics,
                                negotiation_round=negotiation_round + 1,
                            )
                            self._record_staff_turn('test_scheduling', pitch)
                            negotiation_round += 1
                            tries += 1
                            if tries > max_inferences:
                                raise TurnLimitReached
                            continue

                        elif action == 'auto':
                            # Free win (gpc == inf): the patient concedes nothing, so skip persuasion
                            negotiation_done = True
                            negotiation_outcome = 'auto'
                            pred_schedule = self._throughput_pred_from_metrics(
                                negotiation_metrics, doctor_information,
                                staff_known_data['department'], staff_known_data['attending_physician'],
                            )
                            reply = self._render_staff_reply(
                                {'type': 'tool', 'result': pred_schedule},
                                'test_scheduling', gt_patient_condition, staff_known_data, natural_express, _prev_suggested_schedule,
                            )
                            self._record_staff_turn('test_scheduling', Information(response=reply, agent=self._chief_agent_name))
                            break

                    # Naive reply turn
                    if prediction['type'] == 'text':
                        self._record_staff_turn('test_scheduling', output)

                    # Record this turn's retrieval outcome
                    elif prediction['type'] == 'tool':
                        if prediction.get('tmp_flag') == 'retrieve':
                            result_dict = self._retrieval_result(
                                prediction, 'test_retrieve', STATUS_CODES['test_retrieve']['identify']
                            )
                            raise DataNotFoundError(colorstr("red", "Error: Patient information not found error."))

                        # Record the staff utterance (clarification 'text' or test/schedule proposal alike)
                        self._record_staff_turn('test_scheduling', output)

                        # Test retrieval case -> update known data + rebuild agent with test tools
                        if 'test_list' in prediction['result']:
                            result = prediction['result']
                            _patient_info = result['patient_fv']
                            test_list = result['test_list']
                            staff_known_data.update({'patient_fv': _patient_info})
                            staff_known_data.update({'department': _patient_info['department']})
                            staff_known_data.update({'attending_physician': _patient_info['attending_physician']})
                            staff_known_data.update({'test': test_list})
                            required_test_codes = [_test['code'] for _test in test_list]
                            filtered_doctor_information = self.environment.get_doctor_schedule(
                                doctor_information=doctor_information,
                                department=staff_known_data['department'],
                                fhir_integration=self.fhir_integration and doctor_information is None,
                            )
                            filtered_test_device_information = self.environment.get_test_device_schedule(
                                test_device_information=test_device_information,
                                test_code=required_test_codes,
                                fhir_integration=self.fhir_integration and test_device_information is None,
                            )

                            # Rebuild the staff agent so subsequent turns have the test-scheduling tools.
                            tool_calling_agent = self.scheduling_agent.build_agent(
                                rule=self.rules,
                                doctor_info=filtered_doctor_information,
                                patient_schedule_list=patient_info,
                                gt_idx=gt_data[0]['index'],
                                test_device_information=filtered_test_device_information,
                                required_test_codes=required_test_codes,
                            )
                            negotiation_ready = True

                        # A successful test schedule ends the inner dialog loop.
                        elif 'test_schedule' in prediction['result']:
                            # Wrong-tool guard
                            expected_preference = 'throughput_max' if gt_patient_condition['preference'] == 'indifferent' \
                                else gt_patient_condition['preference']
                            got_preference = prediction['result'].get('preference_type')
                            if got_preference is not None and got_preference != expected_preference:
                                raise ToolSelectionError(colorstr('red',
                                    f"Wrong scheduling tool: expected '{expected_preference}' but got '{got_preference}'."))

                            pred_schedule = prediction['result']
                            
                            # Export negotiation metrics for EVERY patient (even non-negotiated / non-eligible) for later hospital-utility analysis
                            if negotiation_metrics['gpc'] is None:
                                negotiation_metrics = self._calculate_negotiation_metrics(
                                    patient_preferred_schedule=pred_schedule,
                                    filtered_test_device_information=filtered_test_device_information,
                                    preference=gt_patient_condition['preference'],
                                    unavailable=gt_patient_condition['unavailable'],
                                    patient_name=gt_data[0]['patient_fv']['patient'],
                                )
                            break

                    tries += 1
                    if tries > max_inferences:
                        raise TurnLimitReached

                # Record the negotiation outcome + rounds onto the metrics (persisted with the booking).
                # `success` is derived (accepted/forced), so it is logged but not stored.
                negotiation_metrics['negotiation_outcome'] = negotiation_outcome
                negotiation_metrics['negotiation_rounds'] = negotiation_round
                log(colorstr('cyan', f"[negotiation_result] outcome={negotiation_outcome} rounds={negotiation_round}"))

                # Sanity check
                ## No GT case
                if self.sanity_checker is None:
                    status, status_code = True, STATUS_CODES['correct']
                ## GT existing case
                else:
                    # After a successful negotiation the patient was booked into throughput_max, so judge optimality against that objective (not their original preference).
                    check_condition = {**gt_patient_condition, 'preference': 'throughput_max'} if negotiation_done else gt_patient_condition
                    status, status_code = self.sanity_checker.test_schedule_check(
                        prediction=pred_schedule,
                        gt_patient_condition=check_condition,
                        test_device_information=filtered_test_device_information,
                        doctor_information=doctor_information,
                        environment=self.environment,
                        rule=self.rules,
                    )

                if not status:
                    break

                # Preference rejection logic
                booked_preference = 'throughput_max' \
                    if (negotiation_done or gt_data[i]['preference'] == 'indifferent') else gt_data[i]['preference']
                next_pref_differs = (i != len(gt_data) - 1) and \
                    (gt_data[i + 1]['preference'] != booked_preference) and \
                        (gt_data[i + 1]['preference'] != 'indifferent')     # `indifferent` preference conflicts with rejection logic semantically
                if random.random() < preference_reject_prob and next_pref_differs and len(pred_schedule['test_visit_dates']) > 1:
                    preference_reject_prob *= self.preference_rejection_prob_decay
                    _prev_suggested_schedule = deepcopy(pred_schedule)
                ## Non-rejection case
                else:
                    self._closing_patient_turn(
                        'test_scheduling',
                        f"{gt_data[i]['preference']},{gt_data[i]['distance']}",
                        natural_express=natural_express,
                        closing=self.NEGOTIATION_CLOSINGS[negotiation_outcome],
                        **merged_patient_kwargs,
                    )
                    patient_token_stats = self.patient_agent.client.token_usages
                    break

        except Exception as e:
            result_dict = self._resolve_simulation_error(
                e, 'test_scheduling', gt_patient_condition,
                error_codes={
                    TurnLimitReached: STATUS_CODES['simulation'],                  # Ran out of dialogue turns
                    ToolCallingError: STATUS_CODES['test_retrieve']['identify'],
                    ToolSelectionError: STATUS_CODES['tool'],                      # Wrong scheduling tool for preference
                    SchedulingError: STATUS_CODES['format'],
                    AgentSelectionError: STATUS_CODES['agent'],                    # Wrong agent activated
                },
                result_dict=result_dict,
                dialog_only=(DataNotFoundError,),                                  # Patient information not found
            )
            log("Simulation completed.", color=True)
            token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
            self._finish_scheduling_turn('test_scheduling', verbose)
            return doctor_information, test_device_information, result_dict, token_usage


        # Organize the result for the regular success / failure path
        result_dict = {
            'gt': [gt_patient_condition],
            'pred': [pred_schedule],
            'status': [False],
            'status_code': [status_code],
            'dialog': [preprocess_dialog(self.dialog_history['test_scheduling'])]
        }
        if status:
            try:
                fu_slot = pred_schedule['fu_schedule']
                fu_schedule = fu_slot[next(iter(fu_slot))] if fu_slot else None

                # Post-process the schedule format
                for item in pred_schedule['test_schedule']:
                    item['schedule'] = [item.pop('start'), item.pop('end')]

                # What the `restrict` policy blocked for this booking (and what it would have been without)
                protection_record = self._capacity_protection_record(
                    booked_preference, [t['code'] for t in gt_data[i]['test']], test_device_information, gt_data[i].get('unavailable'),
                )

                final_schedule = {
                    'visit_type': 'follow_up_visit',
                    'patient': staff_known_data['patient_fv']['patient'],
                    'attending_physician': staff_known_data['attending_physician'],
                    'department': staff_known_data['department'],
                    'date': fu_schedule['date'] if fu_schedule else None,
                    'schedule': [fu_schedule['start'], fu_schedule['end']] if fu_schedule else None,
                    'patient_intention': staff_known_data['patient_intention'],
                    'preference': 'throughput_max' if negotiation_done else gt_data[i].get('preference'),
                    'urgency': gt_data[i].get('urgency'),   # read back by the urgent capacity protection and evaluation
                    'preferred_doctor': gt_data[i].get('preferred_doctor'),
                    'valid_from': gt_data[i].get('valid_from'),
                    'unavailable': gt_data[i].get('unavailable'),   # kept so later re-runs honor the same constraint
                    'test': pred_schedule['test_schedule'],
                    'idle_waiting_time': pred_schedule['idle_waiting_time'],
                    'negotiation_metrics': negotiation_metrics,
                    'capacity_protection': protection_record,
                    'restrictions': self._restriction_record([t['code'] for t in gt_data[i]['test']]),
                    'last_updated_time': self.environment.current_time
                }
                result_dict['pred'] = [final_schedule]
                result_dict['status'] = [True]
            except Exception:
                result_dict['status_code'] = [STATUS_CODES['format']]
                log('Error while organizing the prediction. Returning a failure result.', level='warning')

        log("Simulation completed.", color=True)
        token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
        self._finish_scheduling_turn('test_scheduling', verbose)
        return doctor_information, test_device_information, result_dict, token_usage


    def test_canceling_simulate(self,
                                gt_idx: Optional[int] = None,
                                doctor_information: Optional[dict] = None,
                                test_device_information: Optional[dict] = None,
                                patient_schedules: Optional[list[dict]] = None,
                                verbose: bool = True,
                                max_inferences: int = 5,
                                patient_kwargs: dict = {},
                                staff_kwargs: dict = {},
                                **kwargs) -> tuple[Optional[dict], Optional[dict], dict]:
        """
        Simulate a multi-turn conversation for cancelling all of a patient's scheduled tests.

        Args:
            gt_idx (Optional[int], optional): Ground-truth index of the follow-up booking to be cancelled. Defaults to None.
            doctor_information (Optional[dict], optional): Doctor information (required without FHIR). Defaults to None.
            test_device_information (Optional[dict], optional): Test device schedules. Defaults to None.
            patient_schedules (Optional[list[dict]], optional): List of patient appointment schedules. Defaults to None.
            verbose (bool, optional): Whether to print conversation logs. Defaults to True.
            max_inferences (int, optional): Maximum number of dialogue turns. Defaults to 5.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent. Defaults to {}.
            staff_kwargs (dict, optional): Unused (cancellation needs no staff kwargs). Defaults to {}.
            **kwargs: Additional keyword arguments passed to the patient agent.

        Returns:
            tuple[Optional[dict], Optional[dict], dict]: Updated doctor information, test device information, and the result dictionary.
        """
        doctor_information, result_dict = self._cancel_simulate(
            'test_cancel',
            gt_idx=gt_idx,
            doctor_information=doctor_information,
            patient_schedules=patient_schedules,
            verbose=verbose,
            max_inferences=max_inferences,
            patient_kwargs=patient_kwargs,
            tool_data={'test_device_information': test_device_information},
            **kwargs,
        )
        return doctor_information, test_device_information, result_dict


    def test_rescheduling_simulate(self,
                                   gt_idx: Optional[int] = None,
                                   doctor_information: Optional[dict] = None,
                                   test_device_information: Optional[dict] = None,
                                   patient_schedules: Optional[list[dict]] = None,
                                   verbose: bool = True,
                                   max_inferences: int = 5,
                                   patient_kwargs: dict = {},
                                   staff_kwargs: dict = {},
                                   **kwargs) -> tuple[Optional[dict], Optional[dict], dict]:
        """
        Simulate a multi-turn conversation for moving a patient's whole test set earlier.

        Args:
            gt_idx (Optional[int], optional): Ground-truth index of the follow-up booking to be rescheduled. Defaults to None.
            doctor_information (Optional[dict], optional): Doctor information (required without FHIR). Defaults to None.
            test_device_information (Optional[dict], optional): Test device schedules. Defaults to None.
            patient_schedules (Optional[list[dict]], optional): List of patient appointment schedules. Defaults to None.
            verbose (bool, optional): Whether to print conversation logs. Defaults to True.
            max_inferences (int, optional): Maximum number of dialogue turns. Defaults to 5.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent. Defaults to {}.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff agent. Defaults to {}.
            **kwargs: Additional keyword arguments passed to the patient and staff agents.

        Returns:
            tuple[Optional[dict], Optional[dict], dict]: Updated doctor information, test device information, and the result dictionary.
        """
        doctor_information, result_dict = self._reschedule_simulate(
            'test_reschedule',
            self._make_test_reschedule_pipeline,
            gt_idx=gt_idx,
            doctor_information=doctor_information,
            patient_schedules=patient_schedules,
            verbose=verbose,
            max_inferences=max_inferences,
            patient_kwargs=patient_kwargs,
            staff_kwargs=staff_kwargs,
            tool_data={'test_device_information': test_device_information},
            **kwargs,
        )
        return doctor_information, test_device_information, result_dict
