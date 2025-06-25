import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Tuple, Optional

from utils.image_preprocess_utils import *



class GPTClient:
    def __init__(self, model: str):
        self._init_environment()
        self.model = model


    def _init_environment(self):
        """
        Initialize OpenAI client.
        """
        load_dotenv(override=True)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", None))

    
    def __make_payload(self,
                      user_prompt: str,
                      system_prompt: Optional[str] = None,
                      image_path: Optional[str] = None,
                      image_size: Optional[Tuple[int]] = None) -> List[dict]:
        """
        Create a payload for API calls to the GPT model.

        Args:
            user_prompt (str): User prompt.
            system_prompt (Optional[str], optional): System prompt. Defaults to None.
            image_path (Optional[str], optional): Image path if you need to send image. Defaults to None.
            image_size (Optional[Tuple[int]], optional): Image size to be resized. Defaults to None.

        Returns:
            List[dict]: Payload including prompts and image data.
        """
        payloads = list()
        user_contents = {"role": "user", "content": []}

        # System prompt that applied globally
        if system_prompt:
            payloads.append(
                {"role": "system", "content": system_prompt}
            )
        
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
                   **kwargs) -> dict:
        """
        Sends a chat completion request to the model with optional image input and system prompt.

        Args:
            user_prompt (str): The main user prompt or query to send to the model.
            system_prompt (Optional[str], optional): An optional system-level prompt to set context or behavior. Defaults to None.
            image_path (Optional[str], optional): Path to an image file to be included in the prompt. Defaults to None.
            image_size (Optional[Tuple[int]], optional): The target image size in (width, height) format, if resizing is needed. Defaults to None.

        Raises:
            FileNotFoundError: If `image_path` is provided but the file does not exist.
            e: Any exception raised during the API call is re-raised.

        Returns:
            dict: The content of the model's response message.
        """
        if image_path and not os.path.exists(image_path):
            raise FileNotFoundError
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.__make_payload(user_prompt, system_prompt, image_path, image_size),
                **kwargs
            )
            return response.choices[0].message.content
        
        except Exception as e:
            raise e
        


if __name__ == "__main__":
    model = "gpt-4o"
    user_prompt = '위 이미지를 아래의 json 형식으로 정리해줘. 만약 메뉴판에 분류가 있을 경우 너가 구분 되도록 파싱해줘\n* Explanation: 음식에 대한 설명을 30자 내외로 간단하게 알려줘.\n* Ingredients: 음식의 주요 성분은 너가 추측해서 알려줘.\n\n형식: ${메뉴명}: {"price": ${가격}, "explanation": ${간단한설명}, "ingredients": ${음식주요성분}}'
    system_prompt = '너는 메뉴판을 보고 메뉴의 종류를 잘 파싱하는 모델이야'
    image_path = 'assets/testset/1.jpg'
    image_size = (960, 960)

    client = GPTClient(model)
    output = client(
        user_prompt, 
        system_prompt, 
        image_path,
        image_size, 
        max_tokens=2000
    )
    print(output)
    