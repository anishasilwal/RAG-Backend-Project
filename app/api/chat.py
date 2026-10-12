from fastapi import APIRouter, HTTPException

from app.schemas import (
    BookingRequest,
    BookingResponse,
    ChatRequest,
    ChatResponse,
)
from app.services.booking import save_booking
from app.services.rag import RAGService


router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chat"],
)


rag_service = RAGService()


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = rag_service.chat(
            session_id=request.session_id,
            question=request.question,
        )

        return ChatResponse(**result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {exc}",
        ) from exc


@router.post(
    "/book",
    response_model=BookingResponse,
)
def book_interview(
    request: BookingRequest,
) -> BookingResponse:
    try:
        booking = save_booking(
            name=request.name,
            email=request.email,
            interview_date=request.interview_date,
            interview_time=request.interview_time,
        )

        return BookingResponse(
            message="Interview booked successfully.",
            booking_id=booking.id,
            name=booking.name,
            email=booking.email,
            interview_date=booking.interview_date,
            interview_time=booking.interview_time,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Booking failed: {exc}",
        ) from exc