import os
from openai import OpenAI
from typing import Any, Optional
from dotenv import load_dotenv, find_dotenv

from h_adminsim.utils import log
from h_adminsim.utils.image_preprocess_utils import (
    encode_image, 
    encode_resize_image, 
    get_image_extension,
)



class GPTClient:
    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        OpenAI chat client that keeps the conversation history and token usage.

        Args:
            model (str): OpenAI model name.
            api_key (Optional[str], optional): OpenAI API key; None reads ``OPENAI_API_KEY`` from the environment/.env. Defaults to None.
        """
        # Iniitialize
        self.model = model
        self._init_environment(api_key)
        self.histories = list()
        self.token_usages = dict()
        self.__first_turn = True


    def _init_environment(self, api_key: Optional[str] = None) -> None:
        """
        Create the OpenAI client.

        Args:
            api_key (Optional[str], optional): OpenAI API key; None reads ``OPENAI_API_KEY`` from the environment/.env. Defaults to None.
        """
        if not api_key:
            dotenv_path = find_dotenv(usecwd=True)
            load_dotenv(dotenv_path, override=True)
            api_key = os.environ.get("OPENAI_API_KEY", None)
        self.client = OpenAI(api_key=api_key)

    
    def reset_history(self, verbose: bool = True) -> None:
        """
        Clear the conversation history and token usage.

        Args:
            verbose (bool, optional): Whether to log the reset. Defaults to True.
        """
        self.__first_turn = True
        self.histories = list()
        self.token_usages = dict()
        if verbose:
            log('Conversation history has been reset.', color=True)

    
    def _make_payload(self,
                      user_prompt: str,
                      image_path: Optional[str] = None,
                      image_size: Optional[tuple[int, int]] = None) -> list[dict]:
        """
        Build the user message for the chat completions API.

        Args:
            user_prompt (str): User prompt.
            image_path (Optional[str], optional): Image to attach (base64 data URL). Defaults to None.
            image_size (Optional[tuple[int, int]], optional): Max ``(width, height)`` to resize the image to. Defaults to None.

        Returns:
            list[dict]: The user message (text and optional image).
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
                 image_size: Optional[tuple[int, int]] = None,
                 using_multi_turn: bool = False,
                 verbose: bool = True,
                 greeting: Optional[str] = None,
                 **kwargs: Any) -> str:
        """
        Send the prompt (with optional image and system prompt) and return the model's reply.
        An empty reply is replaced by 'Could you tell me again?'.

        Args:
            user_prompt (str): User prompt.
            system_prompt (Optional[str], optional): System prompt, added on the first turn. Defaults to None.
            image_path (Optional[str], optional): Image to attach. Defaults to None.
            image_size (Optional[tuple[int, int]], optional): Max ``(width, height)`` to resize the image to. Defaults to None.
            using_multi_turn (bool, optional): Keep the previous history instead of resetting it. Defaults to False.
            verbose (bool, optional): Whether to log history resets. Defaults to True.
            greeting (Optional[str], optional): Assistant greeting added on the first turn. Defaults to None.
            **kwargs (Any): Extra ``chat.completions.create`` options.

        Returns:
            str: The model's response message (stripped).

        Raises:
            FileNotFoundError: If `image_path` is provided but the file does not exist.
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
                    self.histories.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
                
                # Greeting
                if greeting and self.__first_turn:
                    self.histories.append({"role": "assistant", "content": [{"type": "text", "text": greeting}]})
                
                self.__first_turn = False
            
            # User prompt
            self.histories += self._make_payload(user_prompt, image_path, image_size)

            # Model response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.histories,
                **kwargs
            )
            assistant_msg = response.choices[0].message
            if assistant_msg.content == None:
                assistant_msg.content = 'Could you tell me again?'
            assistant_msg.content = assistant_msg.content.strip()
            self.histories.append({"role": assistant_msg.role, "content": [{"type": "text", "text": assistant_msg.content}]})

            # Logging token usage
            if response.usage:
                self.token_usages.setdefault("prompt_tokens", []).append(response.usage.prompt_tokens)
                self.token_usages.setdefault("completion_tokens", []).append(response.usage.completion_tokens)
                self.token_usages.setdefault("total_tokens", []).append(response.usage.total_tokens)
                self.token_usages.setdefault("reasoning_tokens", []).append(response.usage.completion_tokens_details.reasoning_tokens)

            return assistant_msg.content
        
        except Exception:
            raise
