import os
from importlib import resources
from typing import Optional

from h_adminsim.agent import BaseAgent
from h_adminsim.utils import colorstr, log
from h_adminsim.registry import ConversationState



class IntakeAdminStaffAgent(BaseAgent):
    def __init__(self,
                 model: str,
                 department_list: list[str],
                 max_inferences: int = 5,
                 api_key: Optional[str] = None,
                 use_azure: bool = False,
                 use_vertex: bool = False,
                 use_vllm: bool = False,
                 azure_endpoint: Optional[str] = None,
                 vllm_endpoint: Optional[str] = None,
                 system_prompt_path: Optional[str] = None,
                 log_verbose: bool = True,
                 **kwargs):
        
        super().__init__(
            model=model,
            api_key=api_key,
            use_azure=use_azure,
            use_vertex=use_vertex,
            use_vllm=use_vllm,
            azure_endpoint=azure_endpoint,
            vllm_endpoint=vllm_endpoint,
            **kwargs
        )
        
        # Initialize environment
        self.departments = ''.join([f'{i+1}. {department}\n' for i, department in enumerate(department_list)])
        self.current_inference = 0  # Current inference index
        self.max_inferences = max_inferences    # Maximum number of inferences allowed
        
        # Initialize prompt
        self._system_prompt_template = self._init_prompt(system_prompt_path)
        self.build_prompt()
        
        if log_verbose:
            log("Intake adminStaffAgent initialized successfully", color=True)
    

    def _init_prompt(self, system_prompt_path: Optional[str] = None) -> str:
        """
        Initialize the system prompt for the administration staff agent.

        Args:
            system_prompt_path (Optional[str], optional): Path to a custom system prompt file. 
                                                          If not provided, the default system prompt will be used. Defaults to None.

        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.
        """
        # Initialilze with the default system prompt
        if not system_prompt_path:
            prompt_file_name = "opfv_intake_staff_system.txt"
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            system_prompt = file_path.read_text()
        
        # User can specify a custom system prompt
        else:
            if not os.path.exists(system_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {system_prompt_path}"))
            with open(system_prompt_path, 'r') as f:
                system_prompt = f.read()
        return system_prompt


    def build_prompt(self):
        """
        Build the system prompt for the administration office agent using the provided template and patient conditions.
        """
        self.system_prompt = self._system_prompt_template.format(
            total_idx=self.max_inferences,
            curr_idx=self.current_inference,
            remain_idx=self.max_inferences - self.current_inference,
            department=self.departments,
        )
    

    def update_system_prompt(self):
        """
        Identify the current inference round stage and update the system prompt accordingly.
        """
        # First history is index 0, so assign stage 1 instead of 0.
        self.current_inference = max(1, len(list(filter(lambda x: (not isinstance(x, dict) and x.role == 'model') or \
               (isinstance(x, dict) and x.get('role') == 'assistant'), self.client.histories))))
        self.build_prompt()
        if len(self.client.histories) and isinstance(self.client.histories[0], dict) and self.client.histories[0].get('role') == 'system':
            self.client.histories[0]['content'] = self.system_prompt
    

    def act(self, state: ConversationState) -> tuple[str, bool]:
        """
        MAS entry point: handle one intake turn from the shared conversation state.

        Pulls the latest user message from ``state``, calls the agent, and reports
        whether intake is finished (the inference budget is exhausted).

        Args:
            state (ConversationState): Shared conversation state.

        Returns:
            tuple[str, bool]: ``(reply, is_done)``.
        """
        user_prompt = state.messages[-1]["content"] if state.messages else ""
        reply = self(user_prompt)
        is_done = self.current_inference >= self.max_inferences
        return reply, is_done


    def __call__(self,
                 user_prompt: str,
                 using_multi_turn: bool = True,
                 verbose: bool = True,
                 **kwargs) -> str:
        """
        Call the patient agent with a user prompt and return the response.

        Args:
            user_prompt (str): The user prompt to send to the patient agent.
            using_multi_turn (bool, optional): Whether to use multi-turn conversation. Defaults to True.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            str: The response from the patient agent.
        """
        self.update_system_prompt()
        response = self.client(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            using_multi_turn=using_multi_turn,
            greeting=self.staff_greet,     # Only affects the first turn
            verbose=verbose,
            temperature=self.temperature,
            seed=self.random_seed,
            **kwargs
        )
        return response
