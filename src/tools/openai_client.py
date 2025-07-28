from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Tuple, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from registry import ScheduleModel
from utils import log
from utils.image_preprocess_utils import *



class GPTClient:
    def __init__(self, model: str):
        self.model = model
        self._init_environment()
        self.histories = list()
        self._multi_turn_system_prompt_already_set = False


    def _init_environment(self):
        """
        Initialize OpenAI client.
        """
        load_dotenv(override=True)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))

    
    def reset_history(self, keep_system_prompt: bool = False):
        """
        Reset the conversation history.

        Args:
            keep_system_prompt (bool): Whether to retain the system prompt after reset.
                                       Defaults to False.
        """
        system_message = None
        if keep_system_prompt:
            for msg in self.histories:
                if msg["role"] == "system":
                    system_message = msg
                    break
        else:
            self._multi_turn_system_prompt_already_set = False

        self.histories = []
        if system_message:
            self.histories.append(system_message)

    
    def __make_payload(self,
                       user_prompt: str,
                       image_path: Optional[str] = None,
                       image_size: Optional[Tuple[int]] = None) -> List[dict]:
        """
        Create a payload for API calls to the GPT model.

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
                # To ensure the only one system prompt
                if self._multi_turn_system_prompt_already_set and system_prompt:
                    log('Since the initial system prompt was already set, the current system prompt is ignored.', 'warning')
                    system_prompt = None

                # System prompt
                if system_prompt:
                    self.histories.append({"role": "system", "content": system_prompt})
                    self._multi_turn_system_prompt_already_set = True
                
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

            else:
                # To ensure empty history
                self.reset_history()
                
                # System prompt
                payloads = [{"role": "system", "content": system_prompt}] if system_prompt else []
                
                # User prompt
                payloads += self.__make_payload(user_prompt, image_path, image_size)
                
                # Model response
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=payloads,
                    **kwargs
                )
                assistant_msg = response.choices[0].message

            return assistant_msg.content
        
        except Exception as e:
            raise e



class GPTLangChainClient(GPTClient):
    def __init__(self, model: str):
        super(GPTLangChainClient, self).__init__(model)
        self.client_lc = ChatOpenAI(
            model=self.model,
            api_key=self.client.api_key
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
