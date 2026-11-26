import os
import time
from google import genai
from google.genai import types
from typing import Any, Optional
from dotenv import load_dotenv, find_dotenv

from h_adminsim.utils import log
from h_adminsim.utils.common_utils import exponential_backoff
from h_adminsim.utils.image_preprocess_utils import (
    encode_image,
    get_image_extension,
    encode_resize_image,
)



class GeminiClient:
    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Gemini chat client that keeps the conversation history and token usage.

        Args:
            model (str): Gemini model name.
            api_key (Optional[str], optional): Google API key; None reads ``GOOGLE_API_KEY`` from the environment/.env. Defaults to None.
        """
        # Iniitialize
        self.model = model
        self._init_environment(api_key)
        self.histories = list()
        self.token_usages = dict()
        self.__first_turn = True


    def _init_environment(self, api_key: Optional[str] = None) -> None:
        """
        Create the Google Gemini client.

        Args:
            api_key (Optional[str], optional): Google API key; None reads ``GOOGLE_API_KEY`` from the environment/.env. Defaults to None.
        """
        if not api_key:
            dotenv_path = find_dotenv(usecwd=True)
            load_dotenv(dotenv_path, override=True)
            api_key = os.environ.get("GOOGLE_API_KEY", None)
        self.client = genai.Client(api_key=api_key)


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
                      image_size: Optional[tuple[int, int]] = None) -> list[types.Content]:
        """
        Build the user message for the Gemini API.

        Args:
            user_prompt (str): User prompt.
            image_path (Optional[str], optional): Image to attach. Defaults to None.
            image_size (Optional[tuple[int, int]], optional): Max ``(width, height)`` to resize the image to. Defaults to None.

        Returns:
            list[types.Content]: The user content (text and optional image).
        """
        payloads = list()    
        
        # User prompts
        user_content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=user_prompt)]
        )
        
        if image_path:
            bytes_image = encode_resize_image(image_path, image_size, encode_base64=False) if image_size else encode_image(image_path, encode_base64=False)
            extension = 'jpeg' if image_size else get_image_extension(image_path)
            user_content.parts.append(
                types.Part.from_bytes(data=bytes_image, mime_type=f'image/{extension}')
            )
        
        payloads.append(user_content)
        
        return payloads


    def __call__(self,
                 user_prompt: str,
                 system_prompt: Optional[str] = None,
                 image_path: Optional[str] = None,
                 image_size:Optional[tuple[int, int]] = None,
                 using_multi_turn: bool = False,
                 verbose: bool = True,
                 greeting: Optional[str] = None,
                 **kwargs: Any) -> str:
        """
        Send the prompt (with optional image and system prompt) and return the model's reply.
        Empty replies are retried with backoff; after ``max_retry`` (default 5) 'Could you tell me again?' is returned.

        Args:
            user_prompt (str): User prompt.
            system_prompt (Optional[str], optional): System instruction. Defaults to None.
            image_path (Optional[str], optional): Image to attach. Defaults to None.
            image_size (Optional[tuple[int, int]], optional): Max ``(width, height)`` to resize the image to. Defaults to None.
            using_multi_turn (bool, optional): Keep the previous history instead of resetting it. Defaults to False.
            verbose (bool, optional): Whether to log history resets. Defaults to True.
            greeting (Optional[str], optional): Model greeting prepended on the first turn. Defaults to None.
            **kwargs (Any): ``max_retry`` and other ``GenerateContentConfig`` options.

        Returns:
            str: The model's response message.

        Raises:
            FileNotFoundError: If `image_path` is provided but the file does not exist.
        """
        if image_path and not os.path.exists(image_path):
            raise FileNotFoundError
    
        try:
            # To ensure empty history
            if not using_multi_turn:
                self.reset_history(verbose)

            # Greeting
            if greeting and self.__first_turn:
                self.histories.append(types.Content(role='model', parts=[types.Part.from_text(text=greeting)]))
                self.__first_turn = False
            
            # User prompt
            self.histories += self._make_payload(user_prompt, image_path, image_size)

            # System prompt and model response, including handling None cases
            count = 0
            max_retry = kwargs.get('max_retry', 5)
            while 1:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=self.histories,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        **kwargs
                    )
                )

                # Logging token usage
                if response.usage_metadata:
                    prompt_token_cnt = response.usage_metadata.prompt_token_count if isinstance(response.usage_metadata.prompt_token_count, int) else 0
                    candidates_token_cnt = response.usage_metadata.candidates_token_count if isinstance(response.usage_metadata.candidates_token_count, int) else 0
                    total_token_cnt = response.usage_metadata.total_token_count if isinstance(response.usage_metadata.total_token_count, int) else 0
                    thoughts_token_cnt = response.usage_metadata.thoughts_token_count if isinstance(response.usage_metadata.thoughts_token_count, int) else 0
                    self.token_usages.setdefault("prompt_tokens", []).append(prompt_token_cnt)
                    self.token_usages.setdefault("completion_tokens", []).append(candidates_token_cnt)
                    self.token_usages.setdefault("total_tokens", []).append(total_token_cnt)
                    self.token_usages.setdefault("reasoning_tokens", []).append(thoughts_token_cnt)

                # After the maximum retries
                if count >= max_retry:
                    replace_text = 'Could you tell me again?'
                    self.histories.append(types.Content(role='model', parts=[types.Part.from_text(text=replace_text)]))
                    return replace_text
                
                # Exponential backoff logic
                if response.text == None:
                    wait_time = exponential_backoff(count)
                    time.sleep(wait_time)
                    count += 1
                    continue
                else:
                    break

            self.histories.append(types.Content(role='model', parts=[types.Part.from_text(text=response.text)]))
            return response.text
        
        except Exception:
            raise
