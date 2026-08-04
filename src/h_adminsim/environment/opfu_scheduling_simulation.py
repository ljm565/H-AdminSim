import re
import os
import json
import random
from copy import deepcopy
from importlib import resources
from patientsim import PatientAgent
from decimal import Decimal, getcontext
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
from typing import Tuple, Union, Optional, TYPE_CHECKING

from h_adminsim.registry.errors import (
    SchedulingError,
    ToolCallingError,
    DataNotFoundError,
    AgentSelectionError,
)
from h_adminsim.registry import (
    STATUS_CODES,
    SCHEDULE_STATUS,
    OPFU_PREFERENCE_PHRASE_STAFF,
    OPFU_PREFERENCE_PHRASE_PATIENT,
)
from h_adminsim.tools.callback import TokenUsageCallback
from h_adminsim.tools.sanity_checker import SanityChecker
from h_adminsim.tools import SchedulingRule, scheduling_tool_calling
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.common_utils import *

if TYPE_CHECKING:
    from h_adminsim.pipeline import HospitalMAS
    from h_adminsim.agent import SchedulingAdminStaffAgent
    from h_adminsim.environment.hospital import HospitalEnvironment




class OPFUSchedulingSimulation:
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
                 sanity_checker: Optional[SanityChecker] = None):
        
        # Initialize simulation parameters
        getcontext().prec = 10
        self._chief_agent_name = 'follow_up_visit_scheduling'
        self.patient_agent = patient_agent
        self.admin_staff_mas = admin_staff_mas
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
        
    
    @property
    def scheduling_agent(self) -> "SchedulingAdminStaffAgent":
        """
        The scheduling worker (a leaf of the MAS tree).

        Scheduling, cancellation, and rescheduling are all handled by the same
        ``'first_visit_scheduling'`` worker and are independent of the intake flow,
        so it is fetched from the MAS on demand rather than cached as instance state.
        """
        return self.admin_staff_mas.get_agent('follow_up_visit_scheduling')

    
    def _init_prompt(self, schedule_rejection_prompt_path: Optional[str] = None):
        """
        Initialize the schedule rejection system prompt for the administration staff agent.

        Args:
            schedule_rejection_prompt_path (Optional[str], optional): Path to a custom schedule rejection system prompt file. 
                                                                      If not provided, the default system prompt will be used. Defaults to None.

        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.
        """
        # Initialilze with the default system prompt
        if not schedule_rejection_prompt_path:
            prompt_file_name = "opfu_schedule_patient_rejected_system.txt"
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            self.rejection_system_prompt_template = file_path.read_text()
        
        # User can specify a custom system prompt
        else:
            if not os.path.exists(schedule_rejection_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {schedule_rejection_prompt_path}"))
            with open(schedule_rejection_prompt_path, 'r') as f:
                self.rejection_system_prompt_template = f.read()
        
        # Additional prompts for streaming scheduling simulation
        self.patient_satisfaction_system_prompt = (
            "You are a patient looking to schedule an appointment. "
            "Assume that the latest schedule proposed by the hospital administrative staff is satisfactory."
        )
        self.natural_end_phrase = (
            "{schedule}\n"
            "Respond to this suggested schedule, ending with a thank-you nuance.\n"
            "- If the conversation history shows that the same schedule was previously proposed and you rejected it, "
            "respond with resigned understanding that this is likely the earliest available "
            "(e.g., 'Oh, this must be the earliest available. Alright, thank you.'), keeping it under 15 words.\n"
            "- Otherwise, respond with plain satisfaction in 5 words or fewer, "
            "conveying only satisfaction and no dissatisfaction."
        )
        self.patient_evaluation_system_prompt = (
            "You are a patient evaluating whether the proposed appointment meets your scheduling preference. "
            "Your preference is for a specific {preference}: {preferred_condition}."
        )
        self.patient_schedule_evaluation_phrase = (
            "{schedule}\n"
            "Evaluate whether this appointment meets your {preference} preference ({preferred_condition}).\n"
            "- If your preference is 'doctor': accept if the appointment is with {preferred_condition}.\n"
            "- If your preference is 'date': accept if the appointment date is on or after {preferred_condition}.\n"
            "If acceptable, respond with brief acceptance (5 words or fewer) ending with a thank-you, "
            "and include '#ACCEPT' at the very end. "
            "Otherwise, briefly express dissatisfaction and state what you need instead."
        )
        self.end_phrase = "Thank you."


    def _init_agents(self, verbose: bool = True):
        """
        Reset the conversation histories and token usage records of both the Patient and Doctor agents.

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        """
        self.patient_agent.reset_history(verbose=verbose)
        self.admin_staff_mas.reset(verbose=verbose)


    def _init_history(self):
        """
        Reset the dialogue histories.
        """
        self.dialog_history = {
            'test_scheduling': [],
            'test_cancel': [],
            'test_reschedule': [],
        }

    
    def _to_lc_history(self, key: str) -> list:
        """
        Convert the dialog history for the given key into LangChain message objects.

        Args:
            key (str): Key identifying which dialog history to convert.

        Returns:
            list: A list of LangChain HumanMessage and AIMessage objects.
        """
        msgs = []
        for m in self.dialog_history[key]:
            if m["role"] == "Patient":
                msgs.append(HumanMessage(content=m["content"]))
            elif m["role"] == "Staff":
                msgs.append(AIMessage(content=m["content"]))
        return msgs
    

    def _render_staff_reply(self,
                            staff_response: dict,
                            reply_type: str,
                            gt_patient_condition: Optional[dict] = None,
                            staff_known_data: Optional[dict] = None,
                            natural_express: bool = True) -> str:
        """
        Turn a structured staff scheduling result into a natural-language utterance.

        Args:
            staff_response (dict): The result of `scheduling`, either a clarification (``type == 'text'``) or a schedule proposal (``type == 'tool'``).
            reply_type (str): Reply types of the scheduling agent.
            gt_patient_condition (dict): Ground-truth condition for this turn; supplies the GT required-test codes used to flag tests that could not be scheduled.
            staff_known_data (dict): Patient information known to the staff agent; supplies the follow-up doctor.
            natural_express (bool, optional): Whether to phrase a schedule proposal naturally instead of dumping the raw schedule dict. Defaults to True.

        Returns:
            str: The staff utterance to show the patient.
        """
        # Clarification message
        if staff_response['type'] == 'text':
            return staff_response['result']
        
        elif staff_response['type'] == 'tool':
            # Test cancellation confirmation
            if reply_type == 'test_cancel':
                result = staff_response['result']

                # A wrong identification cancels nothing; the loop surfaces it as a failure.
                if result['cancelled_schedule'] is None:
                    return ""

                # Successful cancellation
                cancelled = {k: v for k, v in result['cancelled_schedule'].items()
                             if k in ['patient', 'attending_physician', 'department', 'date', 'test']}
                return f"I've cancelled all your scheduled tests: {cancelled}"

            # Test rescheduling confirmation
            if reply_type == 'test_reschedule':
                tmp_flag = staff_response.get('tmp_flag')
                result = staff_response['result']

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

            if 'test_list' in staff_response['result']:
                test_list = staff_response['result']['test_list']
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
            
            elif 'test_schedule' in staff_response['result']:
                pred_schedule  = staff_response['result']
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
                            f"and can I make an follow-up appointment with {fu_doctor} after {all_results_ready_at}"
                        )
                    else:
                        parts.append(f"(follow-up with {fu_doctor} cannot be booked at this time)")
                
                schedule_summary = "; ".join(parts)
                
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


    def _accumulate_staff_tokens(self,
                                 staff_response: dict,
                                 staff_token_stats: dict,
                                 staff_token_callback: TokenUsageCallback) -> dict:
        """
        Merge this turn's staff token usage into the running stats and return them.

        Args:
            staff_response (dict): The staff scheduling result (carries ``'token'`` for reasoning).
            staff_token_stats (dict): The running per-key token usage to update.
            staff_token_callback (TokenUsageCallback): Cumulative callback for the tool-calling path.

        Returns:
            dict: The updated token statistics.
        """
        if self.scheduling_strategy == 'tool_calling':
            return staff_token_callback.token_usage
        for k, v in staff_response['token'].items():
            if k not in staff_token_stats:
                staff_token_stats[k] = deepcopy(v)
            else:
                staff_token_stats[k].extend(v)
        return staff_token_stats
    

    def _finish_scheduling_turn(self,
                                reply_type: str,
                                verbose: bool = False):
        """
        Hand the floor back to the orchestrator once the scheduling eval loop is done.

        Args:
            reply_type (str): Reply types of the scheduling agent.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        if len(self.admin_staff_mas.path) <= 1:
            return
        
        closing = self.dialog_history[reply_type][-1]['content']
        messages = self.admin_staff_mas.state.messages
        
        # When end abnormally
        if messages and messages[-1]['content'] == closing:
            return
        
        # When end normally
        self.admin_staff_mas.chat(
            user_prompt=closing,
            using_multi_turn=False,
            verbose=verbose,
            is_done=True,
        )
    
    
    @staticmethod
    def postprocessing(strategy: str,
                       data: Union[str, dict],
                       filtered_doctor_information: Optional[dict] = None,
                       required_tests: Optional[list] = None,
                       filtered_test_device_information: Optional[dict] = None,
                       utc_offset: Optional[str] = None,
                       rule: Optional[SchedulingRule] = None,
                       attending_physician: Optional[str] = None) -> Union[str, dict]:
        """
        Attempts to parse the given text as JSON. If parsing succeeds, returns a dictionary;
        otherwise, returns the original string.

        Args:
            strategy (str): Scheduling strategy. It must be either `reasoning` or `tool_calling`.
            data (Union[str, dict]): The text output to post-process, potentially a JSON-formatted string.
            filtered_doctor_information (Optional[dict], optional): Department-filtered doctor information
                                                                    to postprocess the schedule by tool_calling strategy.
            required_tests (Optional[list], optional): Required-test metadata used by the reasoning branch
                                                       to enrich each test_schedule entry.
            filtered_test_device_information (Optional[dict], optional): Test device information used by the
                                                                         reasoning branch to map device codes
                                                                         back to test codes.
            utc_offset (Optional[str], optional): UTC offset used by the reasoning branch to build ISO timestamps.
            rule (Optional[SchedulingRule], optional): `SchedulingRule` instance used by the reasoning branch to
                                               deterministically compute the follow-up consultation slot
                                               via `physician_filter` + `find_earliest_time`.
            attending_physician (Optional[str], optional): Name of the patient's attending physician; used by
                                                            the reasoning branch when computing `fu_schedule`.

        Returns:
            Union[str, dict]: A dictionary if the text is valid JSON, otherwise the original string.
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
                'test_schedule': [], 
                'test_visit_dates': data['test_visit_dates'], 
                'idle_waiting_time': data['idle_waiting_time'],
                'fu_schedule': None, 
                'all_results_ready_at': data['all_results_ready_at']
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


    def update_patient_system_prompt(self, 
                                     patient_condition: Optional[dict] = None,
                                     rejected_preference: Optional[str] = None,
                                     new_system_prompt: Optional[str] = None):
        """
        Update a system prompt of the patient agent for proposed schedule rejection scenario etc.

        Args:
            patient_condition (Optional[dict], optional): Patient ground-truth condition including current preference.
            rejected_preference (Optional[str], optional): The scheduling preference proposed by the staff agent in the previous turn
                                                           that the patient must explicitly reject.
            new_system_prompt (Optional[str], optional): New system prompt to be updated.
        """
        if patient_condition is not None and rejected_preference is not None:
            # Build new system prompts for rejection scenario.
            preference = patient_condition.get('preference')
            preference_desc = OPFU_PREFERENCE_PHRASE_PATIENT[preference]
            rejected_preference_desc = OPFU_PREFERENCE_PHRASE_STAFF[rejected_preference]
            system_prompt = self.rejection_system_prompt_template.format(
                preference=preference,
                preference_desc=preference_desc,
                rejected_preference=rejected_preference_desc,
                personality=self.patient_agent.personality,
            )
        else:
            system_prompt = new_system_prompt

        # Update new system prompts for rejection scenario
        self.patient_agent.system_prompt = system_prompt
        if len(self.patient_agent.client.histories) and \
            isinstance(self.patient_agent.client.histories[0], dict) and \
                self.patient_agent.client.histories[0].get('role') == 'system':
            self.patient_agent.client.histories[0]['content'][0]['text'] = system_prompt


    def _get_rescheduled_test_result(self,
                                     known_condition: dict,
                                     doctor_information: Optional[dict] = None,
                                     test_device_information: Optional[dict] = None,
                                     **kwargs) -> dict:
        """
        Re-run the whole-test-set scheduling (`throughput_max`/`visit_min`/`stay_min` by the booking's preference,
        with `indifferent` treated as `throughput_max`) to find an improved schedule.

        Args:
            known_condition (dict): The original follow-up booking being rescheduled.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s).
            test_device_information (Optional[dict], optional): Test device information containing device schedules.

        Returns:
            dict: Rescheduled test-set schedule (test-scheduling shape).
        """
        # Sanity check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        required_test_codes = [t['code'] for t in known_condition.get('test') or []]
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

        # The agent picks the follow-up test-scheduling tool (`throughput_max`/`visit_min`/`stay_min`) by the booking's
        # original preference (`indifferent` is routed to `follow_up_throughput_max_test_schedule` as the hospital default)
        _schedule_client = self.scheduling_agent.build_agent(
            rule=self.rules,
            doctor_info=filtered_doctor_information,
            only_schedule_tool=True,
            required_test_codes=required_test_codes,
            test_device_information=filtered_test_device_information,
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
        def _strictly_earlier(new_t, old_t):
            return new_t is not None and old_t is not None and compare_iso_time(old_t, new_t) and new_t != old_t

        # Preference-based improvement: `visit_min` fewer visit dates (then earlier results), `stay_min` less idle
        # waiting between same-day tests, `throughput_max` (default) earlier result-ready time.
        # `indifferent` follows the hospital-friendly default policy, so it uses the same criterion as `throughput_max`.
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
            'preferred_doctor': original_schedule.get('preferred_doctor'),
            'valid_from': original_schedule.get('valid_from'),
            'test': new_schedule['test_schedule'],
            'idle_waiting_time': new_schedule['idle_waiting_time'],
            'last_updated_time': self.environment.current_time,
        }
        return final_schedule


    def _make_test_reschedule_pipeline(self,
                                       doctor_information: Optional[dict] = None,
                                       test_device_information: Optional[dict] = None,
                                       **kwargs):
        """
        Build a callable that runs the post-retrieval rescheduling pipeline:
        _get_rescheduled_test_result -> test_schedule_check -> _check_test_reschedule_validity -> waiting list.

        Args:
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s).
            test_device_information (Optional[dict], optional): Test device information containing device schedules.
            **kwargs: Additional keyword arguments forwarded to the inner scheduling agent.

        Returns:
            Callable[[int, dict], dict]: A pipeline function returning a dict with keys
                'action' ('reschedule' | 'waiting_list' | 'schedule_fail'),
                'new_schedule' (dict | None), and optional 'status_code'.
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
                                      **kwargs):
        """
        Update waiting list availability automatically.

        Args:
            doctor_information (dict): A dictionary containing information about the doctor(s).
            test_device_information (dict): Test device information containing device schedules.

        Yields:
            dict: Updated (or not updated) doctor information, test device information, and a result dictionary.
        """
        # Snapshot the list: a successful reschedule pops the entry from `waiting_list` mid-iteration
        for turn, (idx, original) in enumerate(list(self.environment.waiting_list)):
            if original['status'] == SCHEDULE_STATUS['scheduled'] and original.get('visit_type') == 'follow_up_visit':
                new_schedule = self._get_rescheduled_test_result(
                    known_condition=original,
                    doctor_information=doctor_information,
                    test_device_information=test_device_information,
                    **kwargs
                )

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
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details. Defaults to None.
            test_device_information (Optional[dict], optional): A dictionary containing information about the test devices available. Defaults to None.
            reschedule_flag (bool, optional): Whether this process is rescheduling or not. Defaults to False.
            chat_history (list, optional): Chat history. Defaults to [].

        Raises:
            ToolCallingError: If the agent fails to select or execute a valid scheduling tool.
            TypeError: If the prediction or inputs are of an unsupported type.

        Return
            dict: Scheduling processed result.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")
            assert test_device_information is not None, colorstr("red", f"Test device information must be provided if you don't use FHIR.")

        # Initialization based on the known condition from the staff
        result_dict = init_result_dict()
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
                    st = res['status']
                    idx = res['index']

                    # Patient information not found case: -> text
                    if st is None and idx['pred'] == -1:
                        prediction['type'] = 'text'
                        prediction['result'] = "Sorry, we couldn't find a matching information. Could you please check your details again (patient and doctor names)?"
                        return prediction

                    if st is None:  # No GT, retrieved
                        result_dict['gt'].append({'test_retrieve': None})
                        result_dict['pred'].append({'test_retrieve': idx['pred']})
                        result_dict['status'].append(None)
                        result_dict['status_code'].append(None)
                    elif st is False:  # GT exists, identification failed
                        result_dict['gt'].append({'test_retrieve': idx['gt']})
                        result_dict['pred'].append({'test_retrieve': idx['pred']})
                        result_dict['status'].append(False)
                        result_dict['status_code'].append(STATUS_CODES['test_retrieve']['identify'])
                        prediction['tmp_flag'] = 'retrieve'
                    else:  # True
                        result_dict['gt'].append({'test_retrieve': idx['gt']})
                        result_dict['pred'].append({'test_retrieve': idx['pred']})
                        result_dict['status'].append(True)
                        result_dict['status_code'].append(STATUS_CODES['correct'])

                    prediction['result_dict'] = result_dict
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
                DAY=self._DAY,
                TESTS=json.dumps(known_condition['test'], indent=2),
                TEST_DEVICES=json.dumps(filtered_test_device_information, indent=2),
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


    def test_canceling(self,
                       client: AgentExecutor,
                       patient_intention: str,
                       chat_history: list = []) -> dict:
        """
        Handle a multi-turn test cancellation request using a tool-calling agent.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            patient_intention (str): The patient's utterance expressing a cancellation request.
            chat_history (list, optional): Chat history. Defaults to [].

        Raises:
            TypeError: If the prediction or inputs are of an unsupported type.

        Returns:
            dict: Cancelling processed result.
        """
        # Initialize
        result_dict = init_result_dict()

        # Invoke
        prediction = scheduling_tool_calling(
            client=client,
            user_prompt=patient_intention,
            history=chat_history,
        )

        # Canceling result
        if prediction['type'] == 'tool':
            if prediction['result']['status'] is None:
                # Tests not found case: -> return: str
                if prediction['result']['index']['pred'] == -1:
                    prediction['type'] = 'text'
                    prediction['result'] = "Sorry, we couldn't find your scheduled tests. Could you please check your details again (patient and doctor names)?"

                # No GT case
                else:
                    result_dict['gt'].append({'cancel': None})
                    result_dict['pred'].append({'cancel': prediction['result']['index']['pred']})
                    result_dict['status'].append(None)
                    result_dict['status_code'].append(None)
                    prediction['result_dict'] = result_dict

                return prediction

            # Retrieval fail case
            elif prediction['result']['status'] is False:
                result_dict['gt'].append({'cancel': prediction['result']['index']['gt']})
                result_dict['pred'].append({'cancel': prediction['result']['index']['pred']})
                result_dict['status'].append(prediction['result']['status'])
                result_dict['status_code'].append(STATUS_CODES['cancel']['identify'])
                prediction['result_dict'] = result_dict
                return prediction

            # Successful cancellation case -> return: dict
            else:
                result_dict['gt'].append({'cancel': prediction['result']['index']['gt']})
                result_dict['pred'].append({'cancel': prediction['result']['index']['pred']})
                result_dict['status'].append(prediction['result']['status'])
                result_dict['status_code'].append(STATUS_CODES['correct'])
                prediction['result_dict'] = result_dict
                return prediction

        # Clarification message case -> return: str
        elif prediction['type'] == 'text':
            return prediction

        # Error
        else:
            raise TypeError(colorstr("red", "Error: Unexpected return type from canceling method."))


    def test_rescheduling(self,
                          client: AgentExecutor,
                          patient_intention: str,
                          chat_history: list = []) -> dict:
        """
        Handle a multi-turn test rescheduling request using a tool-calling agent.

        Args:
            client (AgentExecutor): The agent executor to handle tool calls or conversation.
            patient_intention (str): The patient's utterance expressing a rescheduling request.
            chat_history (list, optional): Chat history. Defaults to [].

        Raises:
            TypeError: If the returned type is not supported.

        Returns:
            dict: Rescheduling processed result.
        """
        # Initialize
        result_dict = init_result_dict()

        # Invoke
        prediction = scheduling_tool_calling(
            client=client,
            user_prompt=patient_intention,
            history=chat_history,
        )

        if prediction['type'] == 'tool':
            res = prediction['result']
            st = res['status']
            idx = res['index']

            # Booking not found case: -> text
            if st is None and idx['pred'] == -1:
                prediction['type'] = 'text'
                prediction['result'] = "Sorry, we couldn't find your scheduled tests. Could you please check your details again (patient and doctor names)?"
                return prediction

            # Build retrieval result_dict
            if st is None:  # No GT, retrieved
                result_dict['gt'].append({'reschedule': None})
                result_dict['pred'].append({'reschedule': idx['pred']})
                result_dict['status'].append(None)
                result_dict['status_code'].append(None)
            elif st is False:  # GT exists, identification failed
                result_dict['gt'].append({'reschedule': idx['gt']})
                result_dict['pred'].append({'reschedule': idx['pred']})
                result_dict['status'].append(False)
                result_dict['status_code'].append(STATUS_CODES['reschedule']['identify'])
                prediction['result_dict'] = result_dict
                prediction['tmp_flag'] = 'retrieve'
                return prediction
            else:  # True
                result_dict['gt'].append({'reschedule': idx['gt']})
                result_dict['pred'].append({'reschedule': idx['pred']})
                result_dict['status'].append(True)
                result_dict['status_code'].append(STATUS_CODES['correct'])

            # Translate pipeline action into tmp_flag + result_dict updates
            action = res.get('action')
            if action == 'reschedule':
                result_dict['pred'] = [res['new_schedule']]
                prediction['tmp_flag'] = 'reschedule'
            elif action == 'waiting_list':
                prediction['tmp_flag'] = 'waiting_list'
            elif action == 'schedule_fail':
                result_dict['pred'] = [res['new_schedule']]
                result_dict['status'] = [False]
                result_dict['status_code'] = [STATUS_CODES['reschedule']['schedule'].format(
                    status_code=res.get('schedule_status_code') or STATUS_CODES['format'])]
                prediction['tmp_flag'] = 'schedule'

            prediction['result_dict'] = result_dict
            return prediction

        # Clarification message case -> return: str
        elif prediction['type'] == 'text':
            return prediction

        # Error
        else:
            raise TypeError(colorstr("red", "Error: Unexpected return type from rescheduling method."))
        

    def test_scheduling_simulate(self,
                                 gt_data: dict,
                                 staff_known_data: dict,
                                 doctor_information: Optional[dict] = None,
                                 test_device_information: Optional[dict] = None,
                                 verbose: bool = False,
                                 max_inferences: int = 8,
                                 natural_express: bool = True,
                                 reasoning_max_tries: int = 0,
                                 patient_kwargs: dict = {},
                                 staff_kwargs: dict = {},
                                 **kwargs) -> Tuple[dict, dict, dict]:
        """
        Simulate a multi-turn outpatient scheduling dialogue between a patient agent and an administrative staff agent.

        Args:
            gt_data (dict): Ground-truth patient condition(s) for each dialogue turn.
            staff_known_data (dict): Patient information known to the staff agent.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s) involved, 
                                                           including availability and other relevant details. Defaults to None.
            test_device_information (Optional[dict], optional): A dictionary containing information about the test devices available. Defaults to None.
            verbose (bool, optional): Whether to log detailed simulation outputs. Defaults to False.
            max_inferences (int, optional): Maximum number of dialogue turns.
            natural_express: (bool, optional): Whether express new schedule as natural or not. Defaults to True.
            reasoning_max_tries (int, optional): Reasoning fallback maximum number of retries. Defaults to 0.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff scheduling function.
            **kwargs: Shared keyword arguments passed to both agents.

        Returns:
            Tuple[dict, dict, dict, dict]: 
                - Doctor information.
                - Test information
                - Result dictionary after scheduling a new appointment.
                - Token statistics.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")
            assert test_device_information is not None, colorstr("red", f"Test device information must be provided if you don't use FHIR.")

        # Initialize agents and result dictionary
        self._init_agents(verbose=verbose)
        staff_token_callback = TokenUsageCallback()
        self._init_history()
        staff_token_stats = {}
        patient_info = self.environment.patient_schedules
        tool_calling_agent = self.scheduling_agent.build_agent(
            rule=self.rules, 
            doctor_info=None,
            patient_schedule_list=patient_info,
            gt_idx=gt_data[0]['index'],
        )
        merged_patient_kwargs = {**patient_kwargs, **kwargs}
        merged_staff_kwargs = {**staff_kwargs, **kwargs}

        # Staff turn closure: captures all of `scheduling`'s simulation-side arguments
        holder = {}
        def staff_turn(user_prompt: str) -> Tuple[str, bool]:
            staff_known_data.update({'patient_intention': user_prompt})
            staff_response = self.test_scheduling(
                tool_calling_agent,
                staff_known_data,
                doctor_information,
                test_device_information,
                chat_history=self._to_lc_history('test_scheduling'),
                reasoning_max_tries=reasoning_max_tries,
                callback=staff_token_callback,
                **merged_staff_kwargs
            )
            holder['response'] = staff_response
            return self._render_staff_reply(
                staff_response, 'test_scheduling', gt_patient_condition, staff_known_data, natural_express
            ), False
        
        # Start conversation
        staff_greet = self.admin_staff_mas.root.agent.staff_greet
        self.dialog_history['test_scheduling'].append({"role": "Staff", "content": staff_greet})
        self.admin_staff_mas.state.messages.append({"role": "Staff", "content": staff_greet})
        log(f"{staff_role(role=self.admin_staff_mas.path[-1].name):<25}: {staff_greet}")

        # Iterate over multiple preferences if exists
        tries = 0
        preference_reject_prob = 0.0 if len(gt_data) <= 1 else self.preference_rejection_prob
        try:
            # Preference iteration
            for i, gt_patient_condition in enumerate(gt_data):
                # For the rejection scenario
                if i != 0:
                    self.update_patient_system_prompt(
                        patient_condition=gt_patient_condition,
                        rejected_preference=gt_data[i-1]['preference']
                    )

                while 1:
                    # Obtain response from patient
                    patient_response = self.patient_agent(
                        self.dialog_history['test_scheduling'][-1]["content"],
                        using_multi_turn=True,
                        verbose=False,
                        **merged_patient_kwargs,
                    )
                    patient_token_stats = self.patient_agent.client.token_usages
                    self.dialog_history['test_scheduling'].append({"role": "Patient", "content": patient_response})
                    role = f"{colorstr('green', 'Patient')} ({gt_patient_condition['preference']})"
                    log(f"{role:<25}: {patient_response}")
                    
                    # Scheduling from staff
                    output = self.admin_staff_mas.chat(
                        user_prompt=patient_response,
                        callback=staff_turn,
                        using_multi_turn=False,
                        verbose=False,
                    )

                    # If wrong agent activated
                    if output.agent != self._chief_agent_name:
                        raise AgentSelectionError(colorstr('red', f'Wrong agent activated, expected {self._chief_agent_name} but got {output.agent}'))

                    staff_response = holder.pop('response')

                    # Token accounting
                    staff_token_stats = self._accumulate_staff_tokens(
                        staff_response, staff_token_stats, staff_token_callback
                    )

                    # Fail to identify the schedule -> surface before recording a staff turn
                    if staff_response['type'] == 'tool' and staff_response.get('tmp_flag') == 'retrieve':
                        result_dict = staff_response['result_dict']
                        raise DataNotFoundError(colorstr("red", "Error: Patient information not found error."))

                    # Record the staff utterance (clarification 'text' or test/schedule proposal alike)
                    rendered_response, _role = output.response, output.agent
                    self.dialog_history['test_scheduling'].append({"role": "Staff", "content": rendered_response})
                    log(f"{staff_role(role=_role):<25}: {rendered_response}")

                    # Advance simulation state based on the structured result
                    if staff_response['type'] == 'tool':
                        # Test retrieval case -> update known data + rebuild agent with test tools
                        if 'test_list' in staff_response['result']:
                            result = staff_response['result']
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

                        # A successful test schedule ends the inner dialog loop.
                        elif 'test_schedule' in staff_response['result']:
                            pred_schedule = staff_response['result']
                            break

                    tries += 1
                    if tries > max_inferences:
                        result_dict = {
                            'gt': [gt_patient_condition],
                            'pred': [None],
                            'status': [False],
                            'status_code': [STATUS_CODES['simulation']],
                            'dialog': [preprocess_dialog(self.dialog_history['test_scheduling'])]
                        }
                        token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
                        self._finish_scheduling_turn('test_scheduling', verbose)
                        return doctor_information, test_device_information, result_dict, token_usage

                # Sanity check
                ## No GT case
                if self.sanity_checker is None:
                    status, status_code = True, STATUS_CODES['correct']
                ## GT existing case
                else:
                    status, status_code = self.sanity_checker.test_schedule_check(
                        prediction=pred_schedule,
                        gt_patient_condition=gt_patient_condition,
                        test_device_information=filtered_test_device_information,
                        doctor_information=doctor_information,
                        environment=self.environment,
                        rule=self.rules,
                    )
                    ###################### TMP ######################
                    if status and gt_patient_condition['preference'] == 'visit_min':
                        test_codes_list = list({t['code'] for t in gt_patient_condition['test']})
                        optimal = self.rules.schedule_tests('throughput_max', filtered_test_device_information, test_codes_list, 10)
                        opt_ready, visit_min_ready = optimal['all_results_ready_at'], pred_schedule['all_results_ready_at']
                        if opt_ready is not None and compare_iso_time(visit_min_ready, opt_ready):
                            delta_h = (str_to_datetime(visit_min_ready) - str_to_datetime(opt_ready)).total_seconds() / 3600
                            log(colorstr('magenta', f'throughput_max: {opt_ready}, visit_min: {visit_min_ready} (+{delta_h:.1f}h)'))
                    ################################################

                if not status:
                    break

                # Preference rejection logic
                next_pref_differs = (i != len(gt_data) - 1) and \
                    (gt_data[i + 1]['preference'] != gt_data[i]['preference'])
                if random.random() < preference_reject_prob and next_pref_differs and len(pred_schedule['test_visit_dates']) > 1:
                    preference_reject_prob *= self.preference_rejection_prob_decay
                ## Non-rejection case
                else:
                    if natural_express:
                        self.update_patient_system_prompt(
                            new_system_prompt=self.patient_satisfaction_system_prompt
                        )
                        patient_response = self.patient_agent(
                            self.natural_end_phrase.format(schedule=self.dialog_history['test_scheduling'][-1]['content']),
                            using_multi_turn=True,
                            verbose=False,
                            **merged_patient_kwargs,
                        )
                        patient_token_stats = self.patient_agent.client.token_usages

                    else:
                        patient_response = self.end_phrase

                    self.dialog_history['test_scheduling'].append({"role": "Patient", "content": patient_response})
                    role = f"{colorstr('green', 'Patient')} ({gt_data[i]['preference']})"
                    log(f"{role:<25}: {patient_response}")

                    break

        except DataNotFoundError:
            result_dict['dialog'].append(preprocess_dialog(self.dialog_history['test_scheduling']))
            log("Simulation completed.", color=True)
            token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
            self._finish_scheduling_turn('test_scheduling', verbose)
            return doctor_information, test_device_information, result_dict, token_usage
        
        except ToolCallingError:
            status = False
            result_dict = {
                'gt': [gt_patient_condition],
                'pred': [None],
                'status': [status],
                'status_code': [STATUS_CODES['test_retrieve']['identify']],
                'dialog': [preprocess_dialog(self.dialog_history['test_scheduling'])]
            }
            log("Simulation completed.", color=True)
            token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
            self._finish_scheduling_turn('test_scheduling', verbose)
            return doctor_information, test_device_information, result_dict, token_usage

        except SchedulingError:
            status = False
            result_dict = {
                'gt': [gt_patient_condition],
                'pred': [None],
                'status': [status],
                'status_code': [STATUS_CODES['format']],
                'dialog': [preprocess_dialog(self.dialog_history['test_scheduling'])]
            }
            log("Simulation completed.", color=True)
            token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
            self._finish_scheduling_turn('test_scheduling', verbose)
            return doctor_information, test_device_information, result_dict, token_usage
        
        except AgentSelectionError as e:
            log(str(e), level='warning')
            status = False
            result_dict = {
                'gt': [gt_patient_condition],
                'pred': [None],
                'status': [status],
                'status_code': [STATUS_CODES['agent']],
                'dialog': [preprocess_dialog(self.dialog_history['test_scheduling'])]
            }
            log("Simulation completed.", color=True)
            token_usage = {'patient_token': patient_token_stats, 'admin_staff_token': staff_token_stats}
            self._finish_scheduling_turn('test_scheduling', verbose)
            return doctor_information, test_device_information, result_dict, token_usage
        
        # Otherwise
        except Exception as e:
            status = False
            status_code = STATUS_CODES['unexpected'].format(e=e)
            log(status_code, level='warning')
            result_dict = {
                'gt': [gt_patient_condition],
                'pred': [None],
                'status': [status],
                'status_code': [status_code],
                'dialog': [preprocess_dialog(self.dialog_history['test_scheduling'])]
            }
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
                
                prediction = {
                    'visit_type': 'follow_up_visit',
                    'patient': staff_known_data['patient_fv']['patient'],
                    'attending_physician': staff_known_data['attending_physician'],
                    'department': staff_known_data['department'],
                    'date': fu_schedule['date'] if fu_schedule else None,
                    'schedule': [fu_schedule['start'], fu_schedule['end']] if fu_schedule else None,
                    'patient_intention': staff_known_data['patient_intention'],
                    'preference': gt_data[i].get('preference'),
                    'preferred_doctor': gt_data[i].get('preferred_doctor'),
                    'valid_from': gt_data[i].get('valid_from'),
                    'test': pred_schedule['test_schedule'],
                    'idle_waiting_time': pred_schedule['idle_waiting_time'],
                    'last_updated_time': self.environment.current_time
                }
                result_dict['pred'] = [prediction]
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
                           **kwargs) -> Tuple[dict, dict, dict]:
        """
        Simulate a multi-turn conversation for cancelling all of a patient's scheduled tests.

        Args:
            gt_idx (Optional[int], optional): Ground-truth index of the follow-up booking to be cancelled. Defaults to None.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s).
            test_device_information (Optional[dict], optional): Test device information containing device schedules. Defaults to None.
            patient_schedules (Optional[list[dict]], optional): List of patient appointment schedules. Defaults to None.
            verbose (bool, optional): Whether to print conversation logs. Defaults to True.
            max_inferences (int, optional): Maximum number of dialogue turns.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff agent.
            **kwargs: Additional keyword arguments passed to the patient and staff agent.

        Raises:
            DataNotFoundError: Schedule not found error.

        Returns:
            Tuple[dict, dict, dict]: Updated doctor information, test device information, and a result dictionary after cancellation.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        # Initialize agents and result dictionary
        result_dict = init_result_dict()
        self._init_history()
        self._init_agents(verbose=verbose)
        patient_schedules = self.environment.patient_schedules if patient_schedules is None else patient_schedules
        doctor_information = self.environment.get_general_doctor_info_from_fhir() if self.fhir_integration else doctor_information
        tool_calling_agent = self.scheduling_agent.build_agent(
            rule=self.rules,
            doctor_info=doctor_information,
            patient_schedule_list=patient_schedules,
            gt_idx=gt_idx,
            test_device_information=test_device_information,
        )
        merged_patient_kwargs = {**patient_kwargs, **kwargs}

        # Staff turn closure: routes the structured cancellation turn through the MAS.
        holder = {}
        def staff_turn(user_prompt: str) -> Tuple[str, bool]:
            staff_response = self.test_canceling(
                client=tool_calling_agent,
                patient_intention=user_prompt,
                chat_history=self._to_lc_history('test_cancel')
            )
            holder['response'] = staff_response
            return self._render_staff_reply(staff_response, 'test_cancel'), False

        # Start conversation
        staff_greet = self.admin_staff_mas.root.agent.staff_greet
        self.dialog_history['test_cancel'].append({"role": "Staff", "content": staff_greet})
        self.admin_staff_mas.state.messages.append({"role": "Staff", "content": staff_greet})
        log(f"{staff_role(role=self.admin_staff_mas.path[-1].name):<25}: {staff_greet}")

        try:
            for _ in range(max_inferences):
                # Obtain response from patient
                patient_response = self.patient_agent(
                    self.dialog_history['test_cancel'][-1]["content"],
                    using_multi_turn=True,
                    verbose=False,
                    **merged_patient_kwargs
                )
                self.dialog_history['test_cancel'].append({"role": "Patient", "content": patient_response})
                role = f"{colorstr('green', 'Patient')} (cancel)"
                log(f"{role:<25}: {patient_response}")

                # Canceling from staff
                output = self.admin_staff_mas.chat(
                    user_prompt=patient_response,
                    callback=staff_turn,
                    using_multi_turn=False,
                    verbose=False,
                )

                # If wrong agent activated
                if output.agent != self._chief_agent_name:
                    raise AgentSelectionError(colorstr('red', f'Wrong agent activated, expected {self._chief_agent_name} but got {output.agent}'))

                staff_response = holder.pop('response')

                # A wrong identification cancels nothing -> surface as a not-found failure
                if staff_response['type'] == 'tool' and staff_response['result_dict']['status'][0] is False:
                    raise DataNotFoundError(colorstr("red", "Error: Schedule not found error."))

                rendered_response, _role = output.response, output.agent
                self.dialog_history['test_cancel'].append({"role": "Staff", "content": rendered_response})
                log(f"{staff_role(role=_role):<25}: {rendered_response}")

                # Tool calling result -> successful cancellation (a clarification 'text' reply just re-iterates)
                if staff_response['type'] == 'tool':
                    result_dict = staff_response['result_dict']

                    # Final response of patient
                    self.dialog_history['test_cancel'].append({"role": "Patient", "content": self.end_phrase})
                    role = f"{colorstr('green', 'Patient')} (cancel)"
                    log(f"{role:<25}: {self.end_phrase}")

                    result_dict['dialog'].append(preprocess_dialog(self.dialog_history['test_cancel']))
                    break

            # The case without any determination during the simulation
            if not len(result_dict['gt']):
                result_dict = {
                    'gt': [{'cancel': gt_idx}],
                    'pred': [None],
                    'status': [False],
                    'status_code': [STATUS_CODES['cancel']['identify']],
                    'dialog': [preprocess_dialog(self.dialog_history['test_cancel'])]
                }

        # Requested schedule indentification error
        except DataNotFoundError:
            result_dict['dialog'].append(preprocess_dialog(self.dialog_history['test_cancel']))

        # Wrong agent activated
        except AgentSelectionError as e:
            log(str(e), level='warning')
            result_dict = {
                'gt': [{'cancel': gt_idx}],
                'pred': [None],
                'status': [False],
                'status_code': [STATUS_CODES['agent']],
                'dialog': [preprocess_dialog(self.dialog_history['test_cancel'])]
            }

        # Tool calling error
        except TypeError:
            result_dict = {
                'gt': [{'cancel': gt_idx}],
                'pred': [None],
                'status': [False],
                'status_code': [STATUS_CODES['cancel']['type']],
                'dialog': [preprocess_dialog(self.dialog_history['test_cancel'])]
            }

        # Otherwhise
        except Exception as e:
            status_code = STATUS_CODES['unexpected'].format(e=e)
            log(status_code, level='warning')
            result_dict = {
                'gt': [{'cancel': gt_idx}],
                'pred': [None],
                'status': [False],
                'status_code': [status_code],
                'dialog': [preprocess_dialog(self.dialog_history['test_cancel'])]
            }

        log("Simulation completed.", color=True)
        self._finish_scheduling_turn('test_cancel', verbose)
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
                              **kwargs) -> Tuple[dict, dict, dict]:
        """
        Simulate a multi-turn conversation for moving a patient's whole test set earlier.

        Args:
            gt_idx (Optional[int], optional): Ground-truth index of the follow-up booking to be rescheduled. Defaults to None.
            doctor_information (Optional[dict], optional): A dictionary containing information about the doctor(s).
            test_device_information (Optional[dict], optional): Test device information containing device schedules. Defaults to None.
            patient_schedules (Optional[list[dict]], optional): List of patient appointment schedules. Defaults to None.
            verbose (bool, optional): Whether to print conversation logs. Defaults to True.
            max_inferences (int, optional): Maximum number of dialogue turns.
            patient_kwargs (dict, optional): Additional keyword arguments passed to the patient agent.
            staff_kwargs (dict, optional): Additional keyword arguments passed to the staff agent.
            **kwargs: Additional keyword arguments passed to the patient and staff agents.

        Raises:
            TypeError: If the return type from the rescheduling method is unexpected.
            DataNotFoundError: Schedule not found error.
            SchedulingError: Scheduling error.

        Returns:
            Tuple[dict, dict, dict]: Updated doctor information, test device information, and a result dictionary after rescheduling.
        """
        # Sanity Check
        if not self.fhir_integration:
            assert doctor_information is not None, colorstr("red", f"Doctor information must be provided if you don't use FHIR.")

        # Initialize agents and result dictionary
        result_dict = init_result_dict()
        self._init_history()
        self._init_agents(verbose=verbose)
        patient_schedules = self.environment.patient_schedules if patient_schedules is None else patient_schedules
        doctor_information = self.environment.get_general_doctor_info_from_fhir() if self.fhir_integration else doctor_information
        merged_patient_kwargs = {**patient_kwargs, **kwargs}
        merged_staff_kwargs = {**staff_kwargs, **kwargs}
        tool_calling_agent = self.scheduling_agent.build_agent(
            rule=self.rules,
            doctor_info=doctor_information,
            patient_schedule_list=patient_schedules,
            gt_idx=gt_idx,
            test_device_information=test_device_information,
            reschedule_pipeline=self._make_test_reschedule_pipeline(doctor_information, test_device_information, **merged_staff_kwargs),
        )

        # Staff turn closure: routes the structured rescheduling turn through the MAS.
        holder = {}
        def staff_turn(user_prompt: str) -> Tuple[str, bool]:
            staff_response = self.test_rescheduling(
                client=tool_calling_agent,
                patient_intention=user_prompt,
                chat_history=self._to_lc_history('test_reschedule'),
            )
            holder['response'] = staff_response
            return self._render_staff_reply(staff_response, 'test_reschedule'), False

        # Start conversation
        staff_greet = self.admin_staff_mas.root.agent.staff_greet
        self.dialog_history['test_reschedule'].append({"role": "Staff", "content": staff_greet})
        self.admin_staff_mas.state.messages.append({"role": "Staff", "content": staff_greet})
        log(f"{staff_role(role=self.admin_staff_mas.path[-1].name):<25}: {staff_greet}")

        try:
            for _ in range(max_inferences):
                # Obtain response from patient
                patient_response = self.patient_agent(
                    self.dialog_history['test_reschedule'][-1]["content"],
                    using_multi_turn=True,
                    verbose=False,
                    **merged_patient_kwargs
                )
                self.dialog_history['test_reschedule'].append({"role": "Patient", "content": patient_response})
                role = f"{colorstr('green', 'Patient')} (move)"
                log(f"{role:<25}: {patient_response}")

                # Rescheduling from staff
                output = self.admin_staff_mas.chat(
                    user_prompt=patient_response,
                    callback=staff_turn,
                    using_multi_turn=False,
                    verbose=False,
                )

                # If wrong agent activated
                if output.agent != self._chief_agent_name:
                    raise AgentSelectionError(colorstr('red', f'Wrong agent activated, expected {self._chief_agent_name} but got {output.agent}'))

                staff_response = holder.pop('response')

                # Tool-calling failures resolve nothing -> surface them before recording a staff turn.
                if staff_response['type'] == 'tool':
                    tmp_flag = staff_response.get('tmp_flag')
                    if tmp_flag == 'retrieve':
                        result_dict = staff_response['result_dict']
                        raise DataNotFoundError(colorstr("red", "Error: Schedule not found error."))
                    elif tmp_flag == 'schedule':
                        result_dict = staff_response['result_dict']
                        raise SchedulingError(colorstr("red", "Error: Scheduling error."))
                    elif tmp_flag not in ('waiting_list', 'reschedule'):
                        raise TypeError(colorstr("red", "Error: Unexpected return type from rescheduling method."))

                rendered_response, _role = output.response, output.agent
                self.dialog_history['test_reschedule'].append({"role": "Staff", "content": rendered_response})
                log(f"{staff_role(role=_role):<25}: {rendered_response}")

                # Tool calling result -> successful reschedule / waiting-list
                if staff_response['type'] == 'tool':
                    result_dict = staff_response['result_dict']

                    # Final response of patient
                    self.dialog_history['test_reschedule'].append({"role": "Patient", "content": self.end_phrase})
                    role = f"{colorstr('green', 'Patient')} (move)"
                    log(f"{role:<25}: {self.end_phrase}")

                    result_dict['dialog'].append(preprocess_dialog(self.dialog_history['test_reschedule']))
                    break

            # The case without any determination during the simulation
            if not len(result_dict['gt']):
                result_dict = {
                    'gt': [{'reschedule': gt_idx}],
                    'pred': [None],
                    'status': [False],
                    'status_code': [STATUS_CODES['reschedule']['identify']],
                    'dialog': [preprocess_dialog(self.dialog_history['test_reschedule'])]
                }

        # Requested schedule indentification error
        except DataNotFoundError:
            result_dict['dialog'].append(preprocess_dialog(self.dialog_history['test_reschedule']))

        # Scheduling Error:
        except SchedulingError:
            result_dict['dialog'].append(preprocess_dialog(self.dialog_history['test_reschedule']))

        # Wrong agent activated
        except AgentSelectionError as e:
            log(str(e), level='warning')
            result_dict = {
                'gt': [{'reschedule': gt_idx}],
                'pred': [None],
                'status': [False],
                'status_code': [STATUS_CODES['agent']],
                'dialog': [preprocess_dialog(self.dialog_history['test_reschedule'])]
            }

        # Tool calling error
        except TypeError:
            result_dict = {
                'gt': [{'reschedule': gt_idx}],
                'pred': [None],
                'status': [False],
                'status_code': [STATUS_CODES['reschedule']['type']],
                'dialog': [preprocess_dialog(self.dialog_history['test_reschedule'])]
            }

        # Otherwise
        except Exception as e:
            status_code = STATUS_CODES['unexpected'].format(e=e)
            log(status_code, level='warning')
            result_dict = {
                'gt': [{'reschedule': gt_idx}],
                'pred': [None],
                'status': [False],
                'status_code': [status_code],
                'dialog': [preprocess_dialog(self.dialog_history['test_reschedule'])]
            }

        log("Simulation completed.", color=True)
        self._finish_scheduling_turn('test_reschedule', verbose)
        return doctor_information, test_device_information, result_dict
