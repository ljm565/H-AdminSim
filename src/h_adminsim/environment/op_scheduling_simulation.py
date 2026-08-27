import os
from copy import deepcopy
from typing import Callable, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage

from h_adminsim.registry import STATUS_CODES
from h_adminsim.registry.errors import AgentSelectionError
from h_adminsim.tools.callback import TokenUsageCallback
from h_adminsim.environment.op_simulation import OPSimulation
from h_adminsim.utils import log, colorstr
from h_adminsim.utils.common_utils import init_result_dict, preprocess_dialog
from h_adminsim.utils.prompt_utils import load_prompt



class TurnLimitReached(Exception):
    """
    Raised when a simulation loop exhausts `max_inferences` without the staff
    agent producing a schedule, so the shared error tail can record it like any
    other terminal condition instead of returning early from inside the loop.
    """


class OPSchedulingSimulation(OPSimulation):
    """
    Shared mechanics for the scheduling simulations
    (``OPFVSchedulingSimulation``, ``OPFUSchedulingSimulation``).

    Subclasses declare the dialogue histories they keep through ``HISTORY_KEYS``
    and supply their own domain logic (staff-reply rendering, post-processing,
    sanity checks). Everything gathered here is identical across the subclasses.
    """
    # Packaged schedule-rejection system prompt for this visit type; declared by subclasses.
    REJECTION_PROMPT: str


    def _init_prompt(self, schedule_rejection_prompt_path: Optional[str] = None):
        """
        Initialize the patient-side prompts used across the scheduling simulations.

        Args:
            schedule_rejection_prompt_path (Optional[str], optional): Path to a custom schedule
                                                                      rejection system prompt file.
                                                                      If not provided, the subclass's
                                                                      `REJECTION_PROMPT` is used.
                                                                      Defaults to None.

        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.
        """
        # Rejection scenario system prompt: packaged per visit type, overridable by the caller
        if not schedule_rejection_prompt_path:
            self.rejection_system_prompt_template = load_prompt(self.REJECTION_PROMPT)
        else:
            if not os.path.exists(schedule_rejection_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {schedule_rejection_prompt_path}"))
            with open(schedule_rejection_prompt_path, 'r') as f:
                self.rejection_system_prompt_template = f.read()

        # Shared prompts driving the patient's reaction to a proposed schedule
        self.patient_satisfaction_system_prompt = load_prompt('opfvfu_schedule_patient_satisfied_system.txt')
        self.natural_end_phrase = load_prompt('opfvfu_schedule_patient_satisfied_user.txt')
        self.patient_evaluation_system_prompt = load_prompt('opfvfu_schedule_patient_evaluation_system.txt')
        self.patient_schedule_evaluation_phrase = load_prompt('opfvfu_schedule_patient_evaluation_user.txt')
        self.end_phrase = "Thank you."


    def _staff_turn(self,
                    user_prompt: str,
                    respond: Callable[[str], Tuple[str, dict]],
                    force_on_misroute: bool = False) -> Tuple[object, dict]:
        """
        Route one staff turn through the MAS and recover both of its results.

        ``admin_staff_mas.chat`` drives the staff turn through a callback that may only
        return the utterance to show the patient, but the simulation also needs the
        structured prediction behind it. This carries that prediction out of the callback
        so callers get both, and applies the simulation's misrouting policy.

        Throughout the simulations, ``staff_response`` names the utterance shown to the
        patient and ``prediction`` names the structured tool-calling/reasoning result.

        Args:
            user_prompt (str): The patient utterance that opens this turn.
            respond (Callable[[str], Tuple[str, dict]]): Produces ``(staff_response,
                                                          prediction)`` for a prompt.
            force_on_misroute (bool, optional): Whether to deterministically re-run the turn on
                                                 the chief agent when the orchestrator picks the
                                                 wrong one, instead of raising. Demo/streaming
                                                 paths must not surface a misroute, whereas
                                                 evaluation paths record it as a failure.
                                                 Defaults to False.

        Raises:
            AgentSelectionError: If the orchestrator activated the wrong agent and
                                 `force_on_misroute` is False.

        Returns:
            Tuple[object, dict]: The MAS output and the structured prediction.
        """
        holder = {}

        def callback(prompt: str) -> Tuple[str, bool]:
            reply, prediction = respond(prompt)
            holder['prediction'] = prediction
            return reply, False

        output = self.admin_staff_mas.chat(
            user_prompt=user_prompt,
            callback=callback,
            using_multi_turn=False,
            verbose=False,
        )

        if output.agent != self._chief_agent_name:
            if not force_on_misroute:
                raise AgentSelectionError(
                    colorstr('red', f'Wrong agent activated, expected {self._chief_agent_name} but got {output.agent}')
                )
            log(f"Wrong agent ({output.agent}) activated; forcing {self._chief_agent_name}.", level="warning")
            output = self.admin_staff_mas.force_chat(
                self._chief_agent_name,
                user_prompt=user_prompt,
                callback=callback,
                using_multi_turn=False,
                verbose=False,
            )

        return output, holder.pop('prediction')


    def _retrieval_result(self,
                          prediction: dict,
                          key: str,
                          identify_code: str) -> dict:
        """
        Translate a retrieval tool's verdict into a result dictionary.

        The retrieval tools decide correctness themselves — ``gt_idx`` is baked into the
        agent at build time — so nothing is judged here; the verdict the tool already
        returned is only reshaped into the recording format.

        Args:
            prediction (dict): A ``type == 'tool'`` staff result whose ``'result'`` carries
                               the tool's ``'status'`` verdict and ``'index'`` pair.
            key (str): Result-dict key naming the retrieval ('cancel', 'reschedule', ...).
            identify_code (str): Status code recorded when identification failed.

        Returns:
            dict: Result dictionary holding this turn's retrieval outcome.
        """
        status = prediction['result']['status']
        index = prediction['result']['index']

        result_dict = init_result_dict()
        result_dict['gt'].append({key: None if status is None else index['gt']})
        result_dict['pred'].append({key: index['pred']})
        result_dict['status'].append(status)
        result_dict['status_code'].append(
            None if status is None
            else (identify_code if status is False else STATUS_CODES['correct'])
        )
        return result_dict


    def _failure_result(self,
                        key: str,
                        gt,
                        status_code: str) -> dict:
        """
        Build the standard single-entry failure result for a simulation run.

        Args:
            key (str): Dialogue-history key of the running simulation.
            gt: Ground-truth entry to record alongside the failure.
            status_code (str): Status code describing why the run failed.

        Returns:
            dict: Result dictionary in the shape the evaluation pipeline expects.
        """
        return {
            'gt': [gt],
            'pred': [None],
            'status': [False],
            'status_code': [status_code],
            'dialog': [preprocess_dialog(self.dialog_history[key])],
        }


    def _resolve_simulation_error(self,
                                  e: Exception,
                                  key: str,
                                  gt,
                                  error_codes: dict,
                                  result_dict: Optional[dict] = None,
                                  dialog_only: tuple = ()) -> dict:
        """
        Translate an exception that ended a simulation loop into a result dictionary.

        Args:
            e (Exception): The exception that ended the loop.
            key (str): Dialogue-history key of the running simulation.
            gt: Ground-truth entry to record alongside the failure.
            error_codes (dict): ``{exception type: status code}`` for this simulation,
                                 ordered most specific first. Anything not listed falls
                                 through to ``STATUS_CODES['unexpected']``.
            result_dict (Optional[dict], optional): Result accumulated so far; required
                                                     only when `dialog_only` is used.
            dialog_only (tuple, optional): Exception types whose `result_dict` was already
                                            populated by the inner tool-calling step, so the
                                            dialogue is appended instead of replacing it.

        Returns:
            dict: The result dictionary to return from the simulation.
        """
        # The loop already recorded the outcome; only the dialogue is missing.
        if dialog_only and isinstance(e, dialog_only):
            result_dict['dialog'].append(preprocess_dialog(self.dialog_history[key]))
            return result_dict

        for exc, status_code in error_codes.items():
            if isinstance(e, exc):
                if isinstance(e, AgentSelectionError):
                    log(str(e), level='warning')
                return self._failure_result(key, gt, status_code)

        # Unexpected failure: surface the exception text in the status code.
        status_code = STATUS_CODES['unexpected'].format(e=e)
        log(status_code, level='warning')
        return self._failure_result(key, gt, status_code)


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


    def _accumulate_staff_tokens(self,
                                 prediction: dict,
                                 staff_token_stats: dict,
                                 staff_token_callback: TokenUsageCallback) -> dict:
        """
        Merge this turn's staff token usage into the running stats and return them.

        Args:
            prediction (dict): The staff scheduling result (carries ``'token'`` for reasoning).
            staff_token_stats (dict): The running per-key token usage to update.
            staff_token_callback (TokenUsageCallback): Cumulative callback for the tool-calling path.

        Returns:
            dict: The updated token statistics.
        """
        if self.scheduling_strategy == 'tool_calling':
            return staff_token_callback.token_usage
        for k, v in prediction['token'].items():
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
