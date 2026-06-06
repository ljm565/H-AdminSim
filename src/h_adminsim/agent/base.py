from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from patientsim.utils.common_utils import set_seed
from patientsim.client import GeminiVertexClient, GPTAzureClient

from h_adminsim.client import GeminiClient, GPTClient, VLLMClient
from h_adminsim.utils import colorstr, log

if TYPE_CHECKING:
    from h_adminsim.registry import ConversationState



# Sentinel returned by a router's `route` to signal that its whole subtree is
# finished and control should bubble up to the parent router.
ROUTE_DONE = "__done__"



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
    def act(self, state: ConversationState) -> tuple[str, bool]:
        """
        Produce this turn's reply.

        Args:
            state (ConversationState): Shared conversation state. The last user
                message is at ``state.messages[-1]``.

        Returns:
            tuple[str, bool]: ``(reply, is_done)`` where ``is_done`` tells the
                MAS this agent has finished its job so control can return to the
                parent router.
        """
        ...

    def route(self,
              state: ConversationState,
              candidates: list[str],
              descriptions: Optional[dict[str, Optional[str]]] = None) -> Optional[str]:
        """
        Pick a child to delegate to.

        Args:
            state (ConversationState): Shared conversation state.
            candidates (list[str]): Names of the child agents available at this
                node. The MAS injects these per call.
            descriptions (Optional[dict[str, Optional[str]]]): Optional mapping of
                candidate name -> human description, to help routing decisions.

        Returns:
            Optional[str]: A child name to delegate to, ``ROUTE_DONE`` to finish
                this subtree (bubble up), or ``None`` to reply to the user
                directly (via ``act``) before routing.

        The default is leaf/worker behavior: never route.
        """
        return None
