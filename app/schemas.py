from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list[dict]


class BookingRequest(BaseModel):
    name: str = Field(..., min_length=1)
    email: str = Field(..., min_length=3)
    interview_date: str = Field(..., min_length=1)
    interview_time: str = Field(..., min_length=1)


class BookingResponse(BaseModel):
    message: str
    booking_id: int
    name: str
    email: str
    interview_date: str
    interview_time: str