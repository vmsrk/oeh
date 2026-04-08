from typing import Literal, Optional, List
from pydantic import BaseModel

Priority = Literal["High", "Medium", "Low"]
ActionType = Literal["Reply", "Archive", "Delete", "Flag"]
InfoType = Literal["sender_history", "product_details", "previous_tickets"]

class Observation(BaseModel):
    email_text: str
    sender: str
    subject: str
    available_info_types: List[InfoType]
    step_count: int
    max_steps: int = 10

class Action(BaseModel):
    request_info: Optional[InfoType] = None
    submit_priority: Optional[Priority] = None
    submit_action: Optional[ActionType] = None

class Reward(BaseModel):
    value: float

class State(BaseModel):
    task_id: str
    step_count: int
    done: bool
    info_already_requested: List[InfoType]
    final_score: Optional[float] = None