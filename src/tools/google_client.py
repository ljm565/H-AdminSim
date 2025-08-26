import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from typing import List, Tuple, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from registry import ScheduleModel
from utils import log
from utils.common_utils import exponential_backoff
from utils.image_preprocess_utils import *



class GeminiClient:
    def __init__(self, model: str):
        self.model = model
        self._init_environment()
        self.chat = None
        self._multi_turn_chat_already_set = False


    def _init_environment(self):
        """
        Initialize Goolge GCP Gemini client.
        """
        load_dotenv(override=True)
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", None))


    def reset_history(self):
        """
        Reset the conversation history.
        """
        self.chat = None
        self._multi_turn_chat_already_set = False


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
                 **kwargs) -> str:
        """
        Sends a chat completion request to the model with optional image input and system prompt.

        Args:
            user_prompt (str): The main user prompt or query to send to the model.
            system_prompt (Optional[str], optional): An optional system-level prompt to set context or behavior. Defaults to None.
            image_path (Optional[str], optional): Path to an image file to be included in the prompt. Defaults to None.
            image_size (Optional[Tuple[int]], optional): The target image size in (width, height) format, if resizing is needed. Defaults to None.
            using_multi_turn (bool): Whether to structure it as multi-turn. Defaults to False.

        Raises:
            FileNotFoundError: If `image_path` is provided but the file does not exist.
            e: Any exception raised during the API call is re-raised.

        Returns:
            str: The model's response message.
        """
        if image_path and not os.path.exists(image_path):
            raise FileNotFoundError
    
        try:
            if using_multi_turn:
                if self._multi_turn_chat_already_set and system_prompt:
                    log('Since the initial system prompt was already set, the current system prompt is ignored.', 'warning')
                    system_prompt = None

                if not self.chat:
                    self.chat = self.client.chats.create(
                        model=self.model,
                        config=types.GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None,
                    )
                    self._multi_turn_chat_already_set = True
                
                # User prompt and model response
                payloads = self.__make_payload(user_prompt, image_path, image_size)
                response = self.chat.send_message(payloads[0].parts)

            else:
                # To ensure empty history
                self.reset_history()

                # User prompt
                payloads = self.__make_payload(user_prompt, image_path, image_size)
                
                # System prompt and model response, including handling None cases
                count = 0
                retry_count = kwargs.get('retry_count', 5)
                while 1:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=payloads,
                        config=types.GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None,
                        **kwargs
                    )

                    # After the maximum retries
                    if count >= retry_count:
                        break
                    
                    # Exponential backoff logic
                    if response.text == None:
                        wait_time = exponential_backoff(count)
                        time.sleep(wait_time)
                        count += 1
                        continue
                    else:
                        break

            return response.text
        
        except Exception as e:
            raise e



class GeminiLangChainClient(GeminiClient):
    def __init__(self, model: str):
        super(GeminiLangChainClient, self).__init__(model)
        self.client_lc = ChatGoogleGenerativeAI(
            model=self.model,
            api_key=self.client._api_client.api_key
        )

    
    def __call__(self,
                 user_prompt: str,
                 system_prompt: Optional[str] = None,
                 image_path: Optional[str] = None,
                 image_size: Optional[Tuple[int]] = None,
                 using_multi_turn: bool = False,
                 **kwargs) -> str:
        try:
            # To ensure empty history
            self.reset_history()

            # Prompts
            parser = JsonOutputParser(pydantic_object=ScheduleModel)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ('system', system_prompt),
                    ('human', user_prompt)
                ]
            ).partial(format_instructions=parser.get_format_instructions())
            chain = prompt | self.client_lc | parser
            
            # Model response
            response = chain.invoke(kwargs)

            return response
        
        except Exception as e:
            raise e
