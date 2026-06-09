from __future__ import annotations
import re
import json
from abc import ABC, abstractmethod
from typing import Optional, Union
from patientsim.utils.common_utils import set_seed
from patientsim.client import GeminiVertexClient, GPTAzureClient

from h_adminsim.client import GeminiClient, GPTClient, VLLMClient
from h_adminsim.utils import colorstr



class BaseAgent(ABC):
    def __init__(self,
                 model: str,
                 api_key: Optional[str] = None,
                 use_azure: bool = False,
                 use_vertex: bool = False,
                 use_vllm: bool = False,
                 azure_endpoint: Optional[str] = None,
                 vllm_endpoint: Optional[str] = None,
                 **kwargs):
        
        # Initialize base model, environment
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
        self._init_env(**kwargs)
    

    def _init_model(self,
                    model: str,
                    api_key: Optional[str] = None,
                    use_azure: bool = False,
                    use_vertex: bool = False,
                    use_vllm: bool = False,
                    azure_endpoint: Optional[str] = None,
                    vllm_endpoint: Optional[str] = None):
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
        

    def _init_env(self, **kwargs):
        """
        Initialize the environment with default settings.
        """
        self.random_seed = kwargs.get('random_seed', None)
        self.temperature = kwargs.get('temperature', 0.2)   # For various responses. If you want deterministic responses, set it to 0.
        self.staff_greet = kwargs.get('staff_greet', "Hello, how can I help you?")
        self.ROUTE_DONE = "#DONE"
        
        # Set random seed for reproducibility
        if self.random_seed:
            set_seed(self.random_seed)


    def reset_history(self, verbose: bool = True):
        """
        Reset the conversation history.

        Args:
            verbose (bool): Whether to print verbose output. Defaults to True.
        """
        self.client.reset_history(verbose=verbose)


    @abstractmethod
    def act(self, *args, **kwargs) -> tuple[str, bool]:
        """
        Produce this turn's reply as a leaf worker, returning ``(reply, is_done)``.

        Leaf workers (e.g. the intake agent) override this. Routers (e.g. the
        orchestrator) are driven through ``__call__`` and never act as a leaf, so
        the default raises rather than returning a value.
        """
        raise NotImplementedError(
            colorstr("red", f"{type(self).__name__} does not implement act(); it is a router, not a leaf worker.")
        )


    def build_subagent_routings(self,
                                sub_agents: Optional[dict] = None) -> str:
        """
        Build a system prompt for routing among sub-agents based on the provided descriptions.

        Args:
            sub_agents (Optional[dict], optional): A dictionary mapping sub-agent names to their descriptions. Defaults to None.

        Returns:
            str: A formatted string listing the sub-agents and their descriptions.
        """
        sub_agents = sub_agents or {}
        descriptions = {n: c.description for n, c in sub_agents.items()}
        listing = "\n".join(
            f"- {name}: {desc or 'no description provided'}"
            for name, desc in descriptions.items()
        )
        return listing
    

    def postprocessing_json_answer(self, text: str) -> Union[str, dict]:
        """
        Post-processing method of json formatted text output.

        Args:
            text (str): Text input.

        Returns:
            Union[str, dict]: A dictionary if the text is valid JSON, otherwise the original string.
        """
        try:
            if isinstance(text, str):
                match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    text_dict = json.loads(json_str)
                else:
                    try:
                        text_dict = json.loads(text)
                    except:
                        return text
            else:
                text_dict = text
            return text_dict
        except:
            return str(text)
