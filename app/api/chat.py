from fastapi import APIRouter, HTTPException

from app.schemas import (
    ChatRequest,
    ChatResponse,
)
from app.services.graph import ConversationalGraph


router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chat"],
)


conversation_graph = ConversationalGraph()


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = conversation_graph.chat(
            session_id=request.session_id,
            question=request.question,
        )

        return ChatResponse(**result)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {exc}",
        ) from exc