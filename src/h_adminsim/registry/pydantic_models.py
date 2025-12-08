from typing import Optional
from pydantic import BaseModel, Field



class PromptRequest(BaseModel):
    user_prompt: str
    system_prompt: Optional[str] = None



class ScheduleItem(BaseModel):
    start: float = Field(description="Start time of the appointment")
    end: float = Field(description="End time of the appointment")



class ScheduleModel(BaseModel):
    schedule: dict[str, ScheduleItem] = Field(description="Doctor's schedule")
    changed_existing_schedule_list: list = Field(description="List of changed appointments among the exised schedules")
    