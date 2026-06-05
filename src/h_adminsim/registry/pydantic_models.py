from typing import Optional
from pydantic import BaseModel



class PromptRequest(BaseModel):
    user_prompt: str
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None
