import os
from importlib import resources
from typing import Union, Optional

from h_adminsim.agent import BaseAgent
from h_adminsim.utils import colorstr, log



class OrchestratorAgent(BaseAgent):
    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 use_vllm: bool = False,
                 vllm_endpoint: Optional[str] = None,
                 system_prompt_path: Optional[str] = None,
                 log_verbose: bool = True,
                 **kwargs):

        super().__init__(
            model=model,
            api_key=api_key,
            use_vllm=use_vllm,
            vllm_endpoint=vllm_endpoint,
            **kwargs
        )

        # Initialize prompt
        self._init_prompt(
            system_prompt_path=system_prompt_path,
        )
        self._pending_reply = None

        if log_verbose:
            log("OrchestratorAgent initialized successfully", color=True)

    
    def _init_prompt(self,
                     system_prompt_path: Optional[str] = None):
        """
        Initialize the system prompt for the administration staff agent.

        Args:
            system_prompt_path (Optional[str], optional): Path to a custom system prompt file. 
                                                          If not provided, the default system prompt will be used. Defaults to None.

        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.
        """
        # Initialilze with the default routing system prompt
        if not system_prompt_path:
            prompt_file_name = 'orchestrator_system.txt'
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            self.system_prompt = file_path.read_text()
        
        # User can specify a custom routing system prompt
        else:
            if not os.path.exists(system_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {system_prompt_path}"))
            with open(system_prompt_path, 'r') as f:
                self.system_prompt = f.read()


    def act(self,
            user_prompt: str,
            using_multi_turn: bool = True,
            verbose: bool = True,
            is_done: bool = False,
            sub_agents: Optional[dict] = None,
            **kwargs) -> tuple[Union[str, dict], bool]:
        """
        MAS entry point: route among the (unfinished) sub-agents for this turn.

        Args:
            user_prompt (str): The user prompt to route on.
            using_multi_turn (bool, optional): Whether to use multi-turn conversation. Defaults to True.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            is_done (bool, optional): Explicit flag indicating whether the agent action should terminate. Defaults to False.
            sub_agents (Optional[dict], optional): Descriptions of the still-unfinished child agents. Defaults to None.

        Returns:
            tuple[Union[str, dict], bool]: ``(response, is_done)`` — the routing
                dict (or direct reply) and whether the subtree is finished.
        """
        sub_agents = sub_agents or {}
        if not sub_agents:                       # all children completed -> whole subtree done
            return {"route": self.ROUTE_DONE}, True

        response = self(
            user_prompt=user_prompt,
            using_multi_turn=using_multi_turn,
            verbose=verbose,
            sub_agents=sub_agents,
            **kwargs
        )
        if (isinstance(response, dict) and response.get('route') == self.ROUTE_DONE) or \
            all(mas_node.is_complete for mas_node in sub_agents.values()): 
            is_done = True
        return response, is_done


    def __call__(self,
                 user_prompt: str,
                 using_multi_turn: bool = True,
                 verbose: bool = True,
                 sub_agents: Optional[dict] = None,
                 max_retries: int = 3,
                 **kwargs) -> str:
        """
        Call the patient agent with a user prompt and return the response.

        Args:
            user_prompt (str): The user prompt to send to the patient agent.
            using_multi_turn (bool, optional): Whether to use multi-turn conversation. Defaults to True.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            sub_agents (Optional[dict], optional): Optional dictionary of sub-agent descriptions to include in the system prompt. Defaults to None.

        Returns:
            str: The response from the patient agent.
        """
        if sub_agents is not None:
            routers = self.build_subagent_routings(sub_agents)
            system_prompt = self.system_prompt.format(routers=routers)
        else:
            system_prompt = self.system_prompt
        
        tries = 0
        while 1:
            response = self.client(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                using_multi_turn=using_multi_turn,
                greeting=self.staff_greet,     # Only affects the first turn
                verbose=verbose,
                temperature=self.temperature,
                seed=self.random_seed,
                **kwargs
            )
            response = self.postprocessing_json_answer(response)
            if isinstance(response, dict) or tries >= max_retries:
                break
            log(f"Orchestrator response is not a JSON object, retrying... (attempt {tries + 1}/{max_retries})", level='warning')
            tries += 1

        return response