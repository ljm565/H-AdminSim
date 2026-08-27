from copy import deepcopy
from langchain_core.messages import HumanMessage, AIMessage

from h_adminsim.environment.op_simulation import OPSimulation
from h_adminsim.tools.callback import TokenUsageCallback



class OPSchedulingSimulation(OPSimulation):
    """
    Shared mechanics for the scheduling simulations
    (``OPFVSchedulingSimulation``, ``OPFUSchedulingSimulation``).

    Subclasses declare the dialogue histories they keep through ``HISTORY_KEYS``
    and supply their own domain logic (staff-reply rendering, post-processing,
    sanity checks). Everything gathered here is identical across the subclasses.
    """
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
