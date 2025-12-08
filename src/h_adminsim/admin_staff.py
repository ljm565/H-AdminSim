import os
from importlib import resources
from typing import Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent, 
    create_tool_calling_agent, 
)

from h_adminsim.utils import colorstr, log
from h_adminsim.client import GeminiClient, GPTClient, VLLMClient
from h_adminsim.tools import SchedulingRule, create_tools



class AdminStaffAgent:
    def __init__(self,
                 target_task: str,
                 model: str,
                 api_key: Optional[str] = None,
                 use_vllm: bool = False,
                 vllm_endpoint: Optional[str] = None,
                 system_prompt_path: Optional[str] = None,
                 user_prompt_path: Optional[str] = None,
                 tool_calling_prompt_path: Optional[str] = None,
                 reasoning_effort: str = 'low',
                 **kwargs) -> None:
        
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
            reasoning_effort=reasoning_effort
        )
        
        # Initialize prompt
        self.system_prompt, self.user_prompt_template, self.tool_calling_prompt = self._init_prompt(
            system_prompt_path=system_prompt_path, 
            user_prompt_path=user_prompt_path,
            tool_calling_prompt_path=tool_calling_prompt_path
        )
        
        log("Administrative staff agent initialized successfully", color=True)
    

    def _init_env(self, **kwargs) -> None:
        """
        Initialize the environment with default settings.
        """
        assert self.target_task in ['first_outpatient_intake', 'first_outpatient_scheduling'], \
            log(colorstr("red", f"Unsupported target task: {self.target_task}. Supported tasks are 'first_outpatient_intake' and 'first_outpatient_scheduling'."))
        

    def _init_model(self,
                    model: str,
                    api_key: Optional[str] = None,
                    use_vllm: bool = False,
                    vllm_endpoint: Optional[str] = None,
                    reasoning_effort: str = 'low') -> None:
        """
        Initialize the model and API client based on the specified model type.

        Args:
            model (str): The administration office agent model to use.
            api_key (Optional[str], optional): API key for the model. If not provided, it will be fetched from environment variables.
                                               Defaults to None.
            use_vllm (bool): Whether to use vLLM client.
            vllm_endpoint (Optional[str], optional): Path to the vLLM server. Defaults to None.
            reasoning_effort (str, optional): Reasoning effort level for the model. Defaults to 'low'.

        Raises:
            ValueError: If the specified model is not supported.
        """
        if 'gemini' in model.lower():
            self.client = GeminiClient(model, api_key)
            self.reasoning_kwargs = {}
            if reasoning_effort:
                log("'reasoning_effort' is not supported for Gemini models and will be ignored.", level='warning')
        
        elif 'gpt' in model.lower():       # TODO: Support o3, o4 models etc.
            self.client = GPTClient(model, api_key)
            self.reasoning_kwargs = {'reasoning_effort': reasoning_effort} if 'gpt-5' in model.lower() else {}
            if 'gpt-5' not in model.lower() and reasoning_effort:
                log(f"'reasoning_effort' is not supported for {model} model and will be ignored.", level='warning')
        
        elif use_vllm:
            self.client = VLLMClient(model, vllm_endpoint)
            self.reasoning_kwargs = {}
            if reasoning_effort:
                log("'reasoning_effort' is not supported for vLLM models and will be ignored.", level='warning')
        
        else:
            raise ValueError(colorstr("red", f"Unsupported model: {model}. Supported models are 'gemini' and 'gpt'."))
        

    def _init_prompt(self, 
                     system_prompt_path: Optional[str] = None, 
                     user_prompt_path: Optional[str] = None,
                     tool_calling_prompt_path: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
        """
        Initialize the system prompt for the administration staff agent.

        Args:
            system_prompt_path (Optional[str], optional): Path to a custom system prompt file. 
                                                          If not provided, the default system prompt will be used. Defaults to None.
            user_prompt_path (Optional[str], optional): Path to a custom user prompt file. 
                                                        If not provided, the default user prompt will be used. Defaults to None.
            tool_calling_prompt_path (Optional[str], optional): Path to a custom tool calling prompt file. 
                                                                If not provided, the default tool calling prompt will be used. Defaults to None.
        Returns:
            Tuple[str, str, Optional[str]]: The system prompt and user prompt templates.

        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.
        """
        # Initialilze with the default system prompt
        if not system_prompt_path:
            prompt_file_name = 'schedule_task_system.txt'
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
            prompt_file_name = 'schedule_task_user.txt'
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            user_prompt_template = file_path.read_text()
        
        # User can specify a custom user prompt
        else:
            if not os.path.exists(user_prompt_path):
                raise FileNotFoundError(colorstr("red", f"User prompt file not found: {user_prompt_path}"))
            with open(user_prompt_path, 'r') as f:
                user_prompt_template = f.read()

        # Initialilze with the default tool calling prompt
        if not tool_calling_prompt_path:
            prompt_file_name = 'schedule_tool_calling_system.txt'
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            tool_calling_prompt = file_path.read_text()
        
        # User can specify a custom tool calling prompt
        else:
            if not os.path.exists(tool_calling_prompt_path):
                tool_calling_prompt = None
            else:
                with open(tool_calling_prompt_path, 'r') as f:
                    tool_calling_prompt = f.read()

        return system_prompt, user_prompt_template, tool_calling_prompt
    

    def reset_history(self, verbose: bool = True) -> None:
        """
        Reset the conversation history.

        Args:
            verbose (bool): Whether to print verbose output. Defaults to True.
        """
        self.client.reset_history(verbose=verbose)


    def build_agent(self, rule: SchedulingRule, doctor_info: dict) -> AgentExecutor:
        """
        Build a LangChain agent with scheduling tools.

        Args:
            rule (SchedulingRule): An instance of SchedulingRule containing scheduling logic.
            doctor_info (dict): A dictionary containing information about doctors.

        Returns:
            AgentExecutor: A LangChain agent executor with the scheduling tools.
        """
        tools = create_tools(rule, doctor_info)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.tool_calling_prompt),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ])
        
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

        else:
            log('Currently, we have supported only Gemini and GPT API-based models.', 'error')
    
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
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
        kwargs.update(self.reasoning_kwargs)
        response = self.client(
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
            using_multi_turn=using_multi_turn,
            verbose=verbose,
            **kwargs
        )
        return response
    