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
from h_adminsim.agent import BaseAgent
from h_adminsim.utils import colorstr, log
from h_adminsim.registry import ConversationState
from h_adminsim.tools import SchedulingRule, create_tools




class SchedulingAdminStaffAgent(BaseAgent):
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
        
        super().__init__(
            model=model,
            api_key=api_key,
            use_vllm=use_vllm,
            vllm_endpoint=vllm_endpoint,
            **kwargs
        )
        
        # Initialize environment
        self.target_task = target_task
        assert self.target_task in ['first_visit_scheduling', 'follow_up_visit_scheduling'], \
            colorstr("red", f"Unsupported target task: {self.target_task}. Supported tasks are `first_visit_scheduling`, and `follow_up_visit_scheduling`.")

        # Initialize prompt
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
        super()._init_env(**kwargs)
        self.general_greet = kwargs.get('general_greet', "How can I help you?")
        self.appn_greet = kwargs.get('appn_greet', "How would you like to schedule the appointment?")
        self.test_greet = kwargs.get('test_greet', "How would you like to schedule the test?")
        self.schedule_suggestion = kwargs.get('schedule_suggestion', "How about this schedule: {schedule}")
        self.natural_schedule_suggestion = kwargs.get(
            'natural_schedule_suggestion', 
            [
                "Can I schedule an appointment with {doctor} on {date} from {start} to {end}?",
                "Could I book an appointment with {doctor} on {date} from {start} to {end}?",
                "Is it possible to schedule an appointment with {doctor} on {date} from {start} to {end}?",
                "Would it be okay to set up an appointment with {doctor} on {date} from {start} to {end}?",
                "Can we arrange an appointment with {doctor} on {date} from {start} to {end}?",
            ]
        )
        self.test_explanation = kwargs.get('test_explanation', "You should take {test_len} test(s): {test_list}.")
        self.natural_test_explanation = kwargs.get(
            'natural_test_explanation',
            [
                "Before your appointment, you'll need to take {test_len} test(s): {test_list}.",
                "You are scheduled to receive {test_len} test(s) prior to the visit, which are: {test_list}.",
                "Please note that {test_len} test(s) will be required beforehand: {test_list}.",
                "Ahead of your appointment, we'll need to conduct {test_len} test(s): {test_list}.",
                "There are {test_len} test(s) you will need to complete before the appointment: {test_list}.",
            ]
        )
        self.fu_schedule_suggestion = kwargs.get(
            'fu_schedule_suggestion',
            "How about this test schedule: {schedule_summary}"
        )
        self.natural_fu_schedule_suggestion = kwargs.get(
            'natural_fu_schedule_suggestion',
            [
                "Could we book your tests as follows: {schedule_summary}?",
                "Would this test schedule work for you: {schedule_summary}?",
                "Can I schedule your tests like this: {schedule_summary}?",
                "Is this arrangement for your tests okay: {schedule_summary}?",
                "How does this test plan sound: {schedule_summary}?",
            ]
        )


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
            sc_tool_calling_prompt_path (Optional[str], optional): Path to a custom scheduling tool calling prompt file. 
                                                                If not provided, the default scheduling tool calling prompt will be used. Defaults to None.

        Raises:
            FileNotFoundError: If the specified system prompt file does not exist.
        """
        # Initialilze with the default system prompt
        if not system_prompt_path:
            if self.target_task == 'first_visit_scheduling':
                prompt_file_name = 'opfv_schedule_staff_system.txt'
            elif self.target_task == 'follow_up_visit_scheduling':
                prompt_file_name = 'opfu_schedule_staff_system.txt'
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            self.system_prompt = file_path.read_text()
        
        # User can specify a custom system prompt
        else:
            if not os.path.exists(system_prompt_path):
                raise FileNotFoundError(colorstr("red", f"System prompt file not found: {system_prompt_path}"))
            with open(system_prompt_path, 'r') as f:
                self.system_prompt = f.read()

        # Initialilze with the default user prompt for scheduling task
        if not scheduling_user_prompt_path:
            if self.target_task == 'first_visit_scheduling':
                prompt_file_name = 'opfv_schedule_staff_reasoning.txt'
            elif self.target_task == 'follow_up_visit_scheduling':
                prompt_file_name = 'opfu_schedule_staff_reasoning.txt'
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            self.scheduling_user_prompt_template = file_path.read_text()
        
        # User can specify a custom user prompt
        else:
            if not os.path.exists(scheduling_user_prompt_path):
                raise FileNotFoundError(colorstr("red", f"User prompt file not found: {scheduling_user_prompt_path}"))
            with open(scheduling_user_prompt_path, 'r') as f:
                self.scheduling_user_prompt_template = f.read()

        # Initialilze with the default tool calling prompt
        if not tool_calling_prompt_path:
            prompt_file_name = 'opfvfu_schedule_staff_tool_calling.txt'
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            self.tool_calling_prompt = file_path.read_text()
        
        # User can specify a custom tool calling prompt
        else:
            if not os.path.exists(tool_calling_prompt_path):
                raise FileNotFoundError(colorstr("red", f"User prompt file not found: {tool_calling_prompt_path}"))
            with open(tool_calling_prompt_path, 'r') as f:
                self.tool_calling_prompt = f.read()
        
        # Initialilze with the only scheduling tool calling prompt
        if not sc_tool_calling_prompt_path:
            # TODO: Add sc-tool calling prompt in OPFU case
            prompt_file_name = 'opfv_schedule_staff_sc_tool_calling.txt'
            file_path = resources.files("h_adminsim.assets.prompts").joinpath(prompt_file_name)
            self.sc_tool_calling_prompt = file_path.read_text()
        
        # User can specify a custom scheduling tool calling prompt
        else:
            if not os.path.exists(sc_tool_calling_prompt_path):
                raise FileNotFoundError(colorstr("red", f"User prompt file not found: {sc_tool_calling_prompt_path}"))
            with open(sc_tool_calling_prompt_path, 'r') as f:
                self.sc_tool_calling_prompt = f.read()


    def build_agent(self,
                    rule: SchedulingRule,
                    doctor_info: dict,
                    patient_schedule_list: Optional[list[dict]] = None,
                    gt_idx: Optional[int] = None,
                    only_schedule_tool: bool = False,
                    reschedule_pipeline: Optional[callable] = None,
                    filtered_test_device_information: Optional[dict] = None,
                    required_test_codes: Optional[list] = None) -> AgentExecutor:
        """
        Build a LangChain agent with scheduling tools.

        Args:
            rule (SchedulingRule): An instance of SchedulingRule containing scheduling logic.
            doctor_info (dict): A dictionary containing information about doctors. Defaults to None.
            patient_schedule_list (Optional[list[dict]], optional): A list of the patient's scheduled appointments. Defaults to None.
            gt_idx (Optional[int], optional): Ground-truth index of the appointment to be cancelled or rescheduled. Defaults to None.
            only_schedule_tool (bool, optional): Whether use only scheduling tools or not. Defaults to False.
            reschedule_pipeline (Optional[callable], optional): Callable executing the post-retrieval rescheduling pipeline. Defaults to None.
            filtered_test_device_information (Optional[dict], optional): Test/device schedules filtered to the patient's required tests.
                                                                          When provided together with ``required_test_codes`` enables the test-scheduling tools.
            required_test_codes (Optional[list], optional): Codes of the tests the patient must take. Defaults to None.

        Returns:
            AgentExecutor: A LangChain agent executor with the scheduling tools.
        """
        tools = create_tools(
            rule, doctor_info, patient_schedule_list, gt_idx, only_schedule_tool,
            reschedule_pipeline=reschedule_pipeline,
            filtered_test_device_information=filtered_test_device_information,
            required_test_codes=required_test_codes,
        )
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


    def act(self, state: ConversationState) -> tuple[str, bool]:
        """
        MAS entry point: handle one scheduling turn from the shared conversation state.

        Args:
            state (ConversationState): Shared conversation state.

        Returns:
            tuple[str, bool]: ``(reply, is_done)``. Scheduling has no simple turn
                budget, so completion currently defaults to ``False`` (the worker
                keeps the floor); refine with a task-specific done-signal as needed.
        """
        user_prompt = state.messages[-1]["content"] if state.messages else ""
        reply = self(user_prompt)
        return reply, False
        

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
