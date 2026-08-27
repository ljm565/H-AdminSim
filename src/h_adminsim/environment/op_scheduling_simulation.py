from copy import deepcopy
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage

from h_adminsim.registry import STATUS_CODES
from h_adminsim.registry.errors import AgentSelectionError
from h_adminsim.tools.callback import TokenUsageCallback
from h_adminsim.environment.op_simulation import OPSimulation
from h_adminsim.utils import log
from h_adminsim.utils.common_utils import preprocess_dialog



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
