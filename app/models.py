"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, field_validator
from typing import Literal


class RequestInput(BaseModel):
    """Incoming user request with natural language input."""
    user_input: str

    @field_validator("user_input")
    @classmethod
    def user_input_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("'user_input' must not be empty or blank")
        return v


class RequestRecord(BaseModel):
    """Internal record tracking a notification request through its lifecycle."""
    id: str
    user_input: str
    status: Literal["queued", "processing", "sent", "failed"] = "queued"


class ExtractedData(BaseModel):
    """Structured data extracted from the AI response."""
    to: str
    message: str
    type: Literal["email", "sms"]

    @field_validator("to")
    @classmethod
    def to_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("'to' must not be empty")
        return stripped

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("'message' must not be empty")
        return stripped
