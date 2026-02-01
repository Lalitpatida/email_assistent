from pydantic import BaseModel
from typing import Literal

class EmailRequest(BaseModel):
    email_text: str
    task_type: Literal["spam", "summary", "action"]


class EmailResponse(BaseModel):
    task_type: str
    result: str


class SummaryResponse(BaseModel):
    task_type: str = "summary"
    summary: str


class SpamResponse(BaseModel):
    task_type: str = "spam"
    is_spam: str    # "SPAM" or "NOT_SPAM"