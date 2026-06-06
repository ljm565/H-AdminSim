import os
from importlib import resources
from typing import Optional, Tuple

from h_adminsim.agent import BaseAgent
from h_adminsim.utils import colorstr, log



class SupervisorAgent(BaseAgent):
    def __init__(self,
                 target_task: str,
                 model: str,
                 api_key: Optional[str] = None,
                 use_vllm: bool = False,
                 vllm_endpoint: Optional[str] = None,
                 system_prompt_path: Optional[str] = None,
                 user_prompt_path: Optional[str] = None,
                 log_verbose: bool = True,
                 **kwargs):
        
        super().__init__(
            model=model,
            api_key=api_key,
            use_vllm=use_vllm,
            vllm_endpoint=vllm_endpoint,
            **kwargs
        )
        
        # Initialize environment
        self.target_task = target_task
        assert self.target_task in ['first_visit_intake'], \
            colorstr("red", f"Unsupported target task: {self.target_task}. Supported is `first_visit_intake`.")
        
        # Initialize prompt
        self.system_prompt, self.user_prompt_template = self._init_prompt(
            system_prompt_path=system_prompt_path, 
            user_prompt_path=user_prompt_path
        )
        
        if log_verbose:
            log(f"Supervisor agent for {self.target_task} initialized successfully", color=True)
    

    def _init_prompt(self, 
                     system_prompt_path: Optional[str] = None, 
                     user_prompt_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Initialize the system prompt for the administration staff agent.

        Args:
            system_prompt_path (Optional[str], optional): Path to a custom system prompt file. 
                                                          If not provided, the default system prompt will be used. Defaults to None.
            user_prompt_path (Optional[str], optional): Path to a custom user prompt file. 
                                                        If not provided, the default user prompt will be used. Defaults to None.
        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.

        Returns:
            Tuple[str, str]: The system prompt and user prompt templates.
        """
        # Initialilze with the default system prompt
        if not system_prompt_path:
            if self.target_task == "first_visit_intake":
                prompt_file_name = "opfv_intake_supervisor_system.txt"
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            system_prompt = file_path.read_text()
        
        # User can specify a custom system prompt
        else:
            if not os.path.exists(system_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {system_prompt_path}"))
            with open(system_prompt_path, 'r') as f:
                system_prompt = f.read()

        # Initialilze with the default user prompt
        if not user_prompt_path:
            if self.target_task == "first_visit_intake":
                prompt_file_name = "opfv_intake_supervisor_user.txt"
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            user_prompt_template = file_path.read_text()
        
        # User can specify a custom user prompt
        else:
            if not os.path.exists(user_prompt_path):
                raise FileNotFoundError(colorstr("red", f"User prompt file not found: {user_prompt_path}"))
            with open(user_prompt_path, 'r') as f:
                user_prompt_template = f.read()

        return system_prompt, user_prompt_template


    def act(self, state) -> tuple[str, bool]:
        """
        Not a MAS turn-taking agent.

        ``SupervisorAgent`` is a post-hoc extractor invoked directly via
        ``__call__``; it never participates in the MAS routing tree, so ``act``
        is implemented only to satisfy the ``BaseAgent`` abstract contract.
        """
        raise NotImplementedError(
            colorstr("red", "SupervisorAgent is not a MAS agent; call it directly via __call__.")
        )


    def __call__(self,
                 user_prompt: str,
                 using_multi_turn: bool = False,
                 verbose: bool = True,
                 **kwargs) -> str:
        """
        Call the patient agent with a user prompt and return the response.

        Args:
            user_prompt (str): The user prompt to send to the patient agent.
            using_multi_turn (bool, optional): Whether to use multi-turn conversation. Defaults to False.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            str: The response from the patient agent.
        """
        response = self.client(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            using_multi_turn=using_multi_turn,
            verbose=verbose,
            temperature=self.temperature,
            **kwargs
        )
        return response
    