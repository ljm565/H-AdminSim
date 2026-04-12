import os
from importlib import resources
from typing import Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent, 
    create_tool_calling_agent, 
)
from patientsim.utils.common_utils import set_seed
from patientsim.client import GeminiVertexClient, GPTAzureClient

from h_adminsim.utils import colorstr, log
from h_adminsim.tools import SchedulingRule, create_tools
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



class SchedulingAdminStaffAgent:
    def __init__(self,
                 target_task: str,
                 model: str,
                 api_key: Optional[str] = None,
                 use_vllm: bool = False,
                 vllm_endpoint: Optional[str] = None,
                 system_prompt_path: Optional[str] = None,
                 scheduling_user_prompt_path: Optional[str] = None,
                 tool_calling_prompt_path: Optional[str] = None,
                 sc_tool_calling_prompt_path: Optional[str] = None,
                 log_verbose: bool = True,
                 **kwargs):
        
        # Initialize environment
        self.target_task = target_task
        self._init_env(**kwargs)
        
        # Initialize model, API client, and other parameters
        self.model = model
        self._init_model(
            model=self.model,
            api_key=api_key,
            use_vllm=use_vllm,
            vllm_endpoint=vllm_endpoint,
        )
        
        # Initialize prompt
        self.system_prompt, self.scheduling_user_prompt_template, self.tool_calling_prompt, self.sc_tool_calling_prompt = \
            self._init_prompt(
                system_prompt_path=system_prompt_path, 
                scheduling_user_prompt_path=scheduling_user_prompt_path,
                tool_calling_prompt_path=tool_calling_prompt_path,
                sc_tool_calling_prompt_path=sc_tool_calling_prompt_path,
            )
        
        if log_verbose:
            log("Scheduling adminStaffAgent initialized successfully", color=True)
    

    def _init_env(self, **kwargs):
        """
        Initialize the environment with default settings.
        """
        assert self.target_task in ['first_visit_intake', 'first_visit_scheduling', 'follow_up_visit_scheduling'], \
            log(colorstr("red", f"Unsupported target task: {self.target_task}. Supported tasks are `first_visit_intake`, `first_visit_scheduling`, and `follow_up_visit_scheduling`."))

        self.random_seed = kwargs.get('random_seed', None)
        self.temperature = kwargs.get('temperature', 0.2)   # For various responses. If you want deterministic responses, set it to 0.
        self.general_staff_greet = kwargs.get('general_staff_greet', "How can I help you?")
        self.staff_greet = kwargs.get('staff_greet', "How would you like to schedule the appointment?")
        self.staff_suggestion = kwargs.get('staff_suggestion', "How about this schedule: {schedule}")
        self.staff_natural_suggestion = kwargs.get(
            'staff_suggestion', 
            [
                "Can I schedule an appointment with {doctor} on {date} from {start} to {end}?",
                "Could I book an appointment with {doctor} on {date} from {start} to {end}?",
                "Is it possible to schedule an appointment with {doctor} on {date} from {start} to {end}?",
                "Would it be okay to set up an appointment with {doctor} on {date} from {start} to {end}?",
                "Can we arrange an appointment with {doctor} on {date} from {start} to {end}?",
            ]
        )
        
        # Set random seed for reproducibility
        if self.random_seed:
            set_seed(self.random_seed)


    def _init_model(self,
                    model: str,
                    api_key: Optional[str] = None,
                    use_vllm: bool = False,
                    vllm_endpoint: Optional[str] = None):
        """
        Initialize the model and API client based on the specified model type.

        Args:
            model (str): The administration office agent model to use.
            api_key (Optional[str], optional): API key for the model. If not provided, it will be fetched from environment variables.
                                               Defaults to None.
            use_vllm (bool): Whether to use vLLM client.
            vllm_endpoint (Optional[str], optional): Path to the vLLM server. Defaults to None.

        Raises:
            ValueError: If the specified model is not supported.
        """
        if 'gemini' in model.lower():
            self.client = GeminiClient(model, api_key)
        elif 'gpt' in model.lower():       # TODO: Support o3, o4 models etc.
            self.client = GPTClient(model, api_key)
        elif use_vllm:
            self.client = VLLMClient(model, vllm_endpoint)
        else:
            raise ValueError(colorstr("red", f"Unsupported model: {model}. Supported models are 'gemini' and 'gpt'."))
        

    def _init_prompt(self, 
                     system_prompt_path: Optional[str] = None, 
                     scheduling_user_prompt_path: Optional[str] = None,
                     tool_calling_prompt_path: Optional[str] = None,
                     sc_tool_calling_prompt_path: Optional[str] = None) -> Tuple[str, str, str, str]:
        """
        Initialize the system prompt for the administration staff agent.

        Args:
            system_prompt_path (Optional[str], optional): Path to a custom system prompt file. 
                                                          If not provided, the default system prompt will be used. Defaults to None.
            scheduling_user_prompt_path (Optional[str], optional): Path to a custom user prompt file. 
                                                                   If not provided, the default user prompt will be used. Defaults to None.
            tool_calling_prompt_path (Optional[str], optional): Path to a custom tool calling prompt file. 
                                                                If not provided, the default tool calling prompt will be used. Defaults to None.
        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.

        Returns:
            Tuple[str, str, str, str]: The system prompt, user prompt templates, tool calling prompt, and the only scheduling tool calling prompt.
        """
        # Initialilze with the default system prompt
        if not system_prompt_path:
            prompt_file_name = 'opfv_schedule_staff_system.txt'
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            system_prompt = file_path.read_text()
        
        # User can specify a custom system prompt
        else:
            if not os.path.exists(system_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {system_prompt_path}"))
            with open(system_prompt_path, 'r') as f:
                system_prompt = f.read()

        # Initialilze with the default user prompt for scheduling task
        if not scheduling_user_prompt_path:
            prompt_file_name = 'opfv_schedule_staff_reasoning.txt'
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            scheduling_user_prompt_template = file_path.read_text()
        
        # User can specify a custom user prompt
        else:
            if not os.path.exists(scheduling_user_prompt_path):
                raise FileNotFoundError(colorstr("red", f"User prompt file not found: {scheduling_user_prompt_path}"))
            with open(scheduling_user_prompt_path, 'r') as f:
                scheduling_user_prompt_template = f.read()

        # Initialilze with the default tool calling prompt
        if not tool_calling_prompt_path:
            if self.target_task == 'first_visit_scheduling':
                prompt_file_name = 'opfv_schedule_staff_tool_calling.txt'
            elif self.target_task == 'follow_up_visit_scheduling':
                prompt_file_name = 'opfu_schedule_staff_tool_calling.txt'
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            tool_calling_prompt = file_path.read_text()
        
        # User can specify a custom tool calling prompt
        else:
            if not os.path.exists(tool_calling_prompt_path):
                raise FileNotFoundError(colorstr("red", f"User prompt file not found: {tool_calling_prompt_path}"))
            else:
                with open(tool_calling_prompt_path, 'r') as f:
                    tool_calling_prompt = f.read()
        
        # Initialilze with the only scheduling tool calling prompt
        if not sc_tool_calling_prompt_path:
            prompt_file_name = 'opfv_schedule_staff_sc_tool_calling.txt'
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            sc_tool_calling_prompt = file_path.read_text()
        
        # User can specify a custom scheduling tool calling prompt
        else:
            if not os.path.exists(sc_tool_calling_prompt_path):
                raise FileNotFoundError(colorstr("red", f"User prompt file not found: {sc_tool_calling_prompt_path}"))
            else:
                with open(sc_tool_calling_prompt_path, 'r') as f:
                    sc_tool_calling_prompt = f.read()

        return system_prompt, scheduling_user_prompt_template, tool_calling_prompt, sc_tool_calling_prompt
    

    def reset_history(self, verbose: bool = True):
        """
        Reset the conversation history.

        Args:
            verbose (bool): Whether to print verbose output. Defaults to True.
        """
        self.client.reset_history(verbose=verbose)


    def build_agent(self, 
                    rule: SchedulingRule, 
                    doctor_info: dict,
                    patient_schedule_list: Optional[list[dict]] = None,
                    gt_idx: Optional[int] = None,
                    only_schedule_tool: bool = False) -> AgentExecutor:
        """
        Build a LangChain agent with scheduling tools.

        Args:
            rule (SchedulingRule): An instance of SchedulingRule containing scheduling logic.
            doctor_info (dict): A dictionary containing information about doctors. Defaults to None.
            patient_schedule_list (Optional[list[dict]], optional): A list of the patient's scheduled appointments. Defaults to None.
            gt_idx (Optional[int], optional): Ground-truth index of the appointment to be canceled or rescheduled. Defaults to None.
            only_schedule_tool (bool, optional): Whether use only scheduling tools or not. Defaults to False.

        Returns:
            AgentExecutor: A LangChain agent executor with the scheduling tools.
        """
        tools = create_tools(rule, doctor_info, patient_schedule_list, gt_idx, only_schedule_tool)
        tool_calling_prompt = self.sc_tool_calling_prompt if only_schedule_tool else self.tool_calling_prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", tool_calling_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ])
        # Gemini series
        if 'gemini' in self.model.lower():
            llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=0,
            )
            agent = create_tool_calling_agent(
                llm=llm,
                tools=tools,
                prompt=prompt
            )
        # GPT series
        elif 'gpt' in self.model.lower():
            llm = ChatOpenAI(
                model_name=self.model, 
                temperature=0 if not 'gpt-5' in self.model.lower() else 1
            )
            agent = create_openai_tools_agent(
                llm=llm,
                tools=tools,
                prompt=prompt
            )
        # vLLM open sources
        else:
            llm = ChatOpenAI(
                model=self.model,
                temperature=0,
                base_url=f"{self.client.vllm_endpoint}/v1",
            )
            agent = create_openai_tools_agent(
                llm=llm,
                tools=tools,
                prompt=prompt
            )
    
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            stream_runnable=False,
            verbose=False,
            max_iterations=1,
            return_intermediate_steps=True,
        )
        return executor
        

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
        response = self.client(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            using_multi_turn=using_multi_turn,
            verbose=verbose,
            temperature=self.temperature,
            **kwargs
        )
        return response
