import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from registry import ScheduleModel
from utils import log
from utils.image_preprocess_utils import *



class VLLMClient:
    def __init__(self, model: str, vllm_url: str):
        self.model = model
        self.vllm_url = vllm_url
        self._init_environment()
        self.histories = list()
        self.token_usages = dict()
        self.__first_turn = False


    def _init_environment(self):
        """
        Initialize vLLM OpenAI-formatted client.
        """
        load_dotenv(override=True)
        self.client = OpenAI(
            base_url=f'{self.vllm_url}/v1',
            api_key='EMPTY'
        )

    
    def reset_history(self, verbose: bool = True) -> None:
        """
        Reset the conversation history.

        Args:
            verbose (bool): Whether to print verbose output. Defaults to True.
        """
        self.__first_turn = True
        self.histories = list()
        self.token_usages = dict()
        if verbose:
            log('Conversation history has been reset.', color=True)

    
    def __make_payload(self,
                       user_prompt: str,
                       image_path: Optional[str] = None,
                       image_size: Optional[Tuple[int]] = None) -> List[dict]:
        """
        Create a payload for API calls to the model.

        Args:
            user_prompt (str): User prompt.
            image_path (Optional[str], optional): Image path if you need to send image. Defaults to None.
            image_size (Optional[Tuple[int]], optional): Image size to be resized. Defaults to None.

        Returns:
            List[dict]: Payload including prompts and image data.
        """
        payloads = list()
        user_contents = {"role": "user", "content": []}

        # User prompts
        user_contents["content"].append(
            {"type": "text", "text": user_prompt}
        )
        if image_path:
            base64_image = encode_resize_image(image_path, image_size) if image_size else encode_image(image_path)
            extension = 'jpeg' if image_size else get_image_extension(image_path)
            user_contents["content"].append(
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/{extension};base64,{base64_image}"
                    }
                }
            )
        
        payloads.append(user_contents)
        
        return payloads


    def __call__(self,
                 user_prompt: str,
                 system_prompt: Optional[str] = None,
                 image_path: Optional[str] = None,
                 image_size: Optional[Tuple[int]] = None,
                 using_multi_turn: bool = False,
                 verbose: bool = True,
                 **kwargs) -> str:
        """
        Sends a chat completion request to the model with optional image input and system prompt.

        Args:
            user_prompt (str): The main user prompt or query to send to the model.
            system_prompt (Optional[str], optional): An optional system-level prompt to set context or behavior. Defaults to None.
            image_path (Optional[str], optional): Path to an image file to be included in the prompt. Defaults to None.
            image_size (Optional[Tuple[int]], optional): The target image size in (width, height) format, if resizing is needed. Defaults to None.
            using_multi_turn (bool): Whether to structure it as multi-turn. Defaults to False.
            verbose (bool): Whether to print verbose output. Defaults to True.

        Raises:
            FileNotFoundError: If `image_path` is provided but the file does not exist.
            e: Any exception raised during the API call is re-raised.

        Returns:
            str: The model's response message.
        """
        if image_path and not os.path.exists(image_path):
            raise FileNotFoundError
        
        try:
            # To ensure empty history
            if not using_multi_turn:
                self.reset_history(verbose)

            if self.__first_turn:
                # System prompt
                if system_prompt:
                    self.histories.append({"role": "system", "content": system_prompt})
                self.__first_turn = False

            # User prompt
            self.histories += self.__make_payload(user_prompt, image_path, image_size)
            
            # Model response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.histories,
                **kwargs
            )
            assistant_msg = response.choices[0].message
            self.histories.append({"role": assistant_msg.role, "content": assistant_msg.content})

            # Logging token usage
            if response.usage:
                self.token_usages.setdefault("prompt_tokens", []).append(response.usage.prompt_tokens)
                self.token_usages.setdefault("completion_tokens", []).append(response.usage.completion_tokens)
                self.token_usages.setdefault("total_tokens", []).append(response.usage.total_tokens)

            return assistant_msg.content
        
        except Exception as e:
            raise e
