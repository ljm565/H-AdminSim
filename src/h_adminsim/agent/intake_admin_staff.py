import os
from importlib import resources
from typing import Optional
from patientsim.utils.common_utils import set_seed
from patientsim.client import GeminiVertexClient, GPTAzureClient

from h_adminsim.utils import colorstr, log
from h_adminsim.client import GeminiClient, GPTClient, VLLMClient



class IntakeAdminStaffAgent:
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
                 **kwargs) -> None:
        
        # Initialize environment
        self.departments = ''.join([f'{i+1}. {department}\n' for i, department in enumerate(department_list)])
        self.current_inference = 0  # Current inference index
        self.max_inferences = max_inferences    # Maximum number of inferences allowed
        self._init_env(**kwargs)
        
        # Initialize model, API client, and other parameters
        self.model = model
        self._init_model(
            model=self.model,
            api_key=api_key,
            use_azure=use_azure,
            use_vertex=use_vertex,
            use_vllm=use_vllm,
            azure_endpoint=azure_endpoint,
            vllm_endpoint=vllm_endpoint
        )
        
        # Initialize prompt
        self._system_prompt_template = self._init_prompt(system_prompt_path)
        self.build_prompt()
        
        if log_verbose:
            log("Intake adminStaffAgent initialized successfully", color=True)
    

    def _init_env(self, **kwargs) -> None:
        """
        Initialize the environment with default settings.
        """
        self.random_seed = kwargs.get('random_seed', None)
        self.temperature = kwargs.get('temperature', 0.2)   # For various responses. If you want deterministic responses, set it to 0.
        self.staff_greet = kwargs.get('staff_greet', "Hello, how can I help you?")
        
        # Set random seed for reproducibility
        if self.random_seed:
            set_seed(self.random_seed)


    def _init_model(self,
                    model: str,
                    api_key: Optional[str] = None,
                    use_azure: bool = False,
                    use_vertex: bool = False,
                    use_vllm: bool = False,
                    azure_endpoint: Optional[str] = None,
                    vllm_endpoint: Optional[str] = None) -> None:
        """
        Initialize the model and API client based on the specified model type.

        Args:
            model (str): The administration office agent model to use.
            api_key (Optional[str], optional): API key for the model. If not provided, it will be fetched from environment variables.
                                               Defaults to None.
            use_azure (bool): Whether to use Azure OpenAI client.
            use_vertex (bool): Whether to use Google Vertex AI client.
            use_vllm (bool): Whether to use vLLM client.
            azure_endpoint (Optional[str], optional): Azure OpenAI endpoint. Defaults to None.
            vllm_endpoint (Optional[str], optional): Path to the vLLM server. Defaults to None.

        Raises:
            ValueError: If the specified model is not supported.
        """
        if 'gemini' in self.model.lower():
            self.client = GeminiVertexClient(model, api_key) if use_vertex else GeminiClient(model, api_key)
        elif 'gpt' in self.model.lower():       # TODO: Support o3, o4 models etc.
            self.client = GPTAzureClient(model, api_key, azure_endpoint) if use_azure else GPTClient(model, api_key)
        elif use_vllm:
            self.client = VLLMClient(model, vllm_endpoint)
        else:
            raise ValueError(colorstr("red", f"Unsupported model: {self.model}. Supported models are 'gemini' and 'gpt'."))
        

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
    
    
    def reset_history(self, verbose: bool = True) -> None:
        """
        Reset the conversation history.

        Args:
            verbose (bool): Whether to print verbose output. Defaults to True.
        """
        self.client.reset_history(verbose=verbose)
    
    
    def build_prompt(self) -> None:
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
