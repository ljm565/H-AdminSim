import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Optional

from h_adminsim.utils import log
from h_adminsim.utils.common_utils import exponential_backoff
from h_adminsim.utils.image_preprocess_utils import *


########### For langchain integration (currently not used) ############
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from h_adminsim.registry import ScheduleModel
#######################################################################



class GeminiClient:
    def __init__(self, model: str, api_key: Optional[str] = None):
        # Iniitialize
        self.model = model
        self._init_environment(api_key)
        self.histories = list()
        self.token_usages = dict()


    def _init_environment(self, api_key: Optional[str] = None):
        """
        Initialize Goolge GCP Gemini client.

        Args:
            api_key (Optional[str]): API key for OpenAI. If not provided, it will
                                     be loaded from environment variables.
        """
        if not api_key:
            dotenv_path = find_dotenv(usecwd=True)
            load_dotenv(dotenv_path, override=True)
            api_key = os.environ.get("GOOGLE_API_KEY", None)
        self.client = genai.Client(api_key=api_key)


    def reset_history(self, verbose: bool = True):
        """
        Reset the conversation history.

        Args:
            verbose (bool): Whether to print verbose output. Defaults to True.
        """
        self.histories = list()
        self.token_usages = dict()
        if verbose:
            log('Conversation history has been reset.', color=True)


    def __make_payload(self,
                       user_prompt: str,
                       image_path: Optional[str] = None,
                       image_size: Optional[Tuple[int]] = None) -> List[types.Content]:
        """
        Create a payload for API calls to the Gemini model.

        Args:
            user_prompt (str): User prompt.
            image_path (Optional[str], optional): Image path if you need to send image. Defaults to None.
            image_size (Optional[Tuple[int]], optional): Image size to be resized. Defaults to None.

        Returns:
            List[types.Content]: Payload including prompts and image data.
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
                 image_size:Optional[Tuple[int]] = None,
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
            
            # User prompt
            self.histories += self.__make_payload(user_prompt, image_path, image_size)

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
                    self.token_usages.setdefault("prompt_tokens", []).append(response.usage_metadata.prompt_token_count)
                    self.token_usages.setdefault("completion_tokens", []).append(response.usage_metadata.candidates_token_count)
                    self.token_usages.setdefault("total_tokens", []).append(response.usage_metadata.total_token_count)

                # After the maximum retries
                if count >= max_retry:
                    replace_text = 'None'
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
        
        except Exception as e:
            raise e



# class GeminiLangChainClient(GeminiClient):
#     def __init__(self, model: str):
#         super(GeminiLangChainClient, self).__init__(model)
#         self.client_lc = ChatGoogleGenerativeAI(
#             model=self.model,
#             api_key=self.client._api_client.api_key
#         )

    
#     def __call__(self,
#                  user_prompt: str,
#                  system_prompt: Optional[str] = None,
#                  image_path: Optional[str] = None,
#                  image_size: Optional[Tuple[int]] = None,
#                  using_multi_turn: bool = False,
#                  **kwargs) -> str:
#         try:
#             # To ensure empty history
#             self.reset_history()

#             # Prompts
#             parser = JsonOutputParser(pydantic_object=ScheduleModel)
#             prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ('system', system_prompt),
#                     ('human', user_prompt)
#                 ]
#             ).partial(format_instructions=parser.get_format_instructions())
#             chain = prompt | self.client_lc | parser
            
#             # Model response
#             response = chain.invoke(kwargs)

#             return response
        
#         except Exception as e:
#             raise e
