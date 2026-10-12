from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.services.booking import save_booking
from app.services.embeddings import EmbeddingService
from app.services.memory import ConversationMemory
from app.services.vector_store import VectorStore


# ---------------------------------------------------------
# Graph State
# ---------------------------------------------------------

class GraphState(TypedDict, total=False):
    session_id: str
    question: str

    route: str
    answer: str

    name: str | None
    email: str | None
    interview_date: str | None
    interview_time: str | None

    booking_id: int | None


# ---------------------------------------------------------
# Structured output for Router
# ---------------------------------------------------------

class RouterDecision(BaseModel):
    route: str = Field(
        description="The user's intent. Must be either 'rag' or 'booking'."
    )


# ---------------------------------------------------------
# Structured output for Booking extraction
# ---------------------------------------------------------

class BookingDetails(BaseModel):
    name: str | None = None
    email: str | None = None
    interview_date: str | None = None
    interview_time: str | None = None


# ---------------------------------------------------------
# Conversational Graph
# ---------------------------------------------------------

class ConversationalGraph:
    def __init__(self) -> None:
        if not settings.llm_api_key:
            raise ValueError("Configure LLM_API_KEY in .env")

        if not settings.llm_model:
            raise ValueError("Configure LLM_MODEL in .env")

        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.memory = ConversationMemory()

        self.graph = self._build_graph()

    # -----------------------------------------------------
    # Build LangGraph
    # -----------------------------------------------------

    def _build_graph(self):
        builder = StateGraph(GraphState)

        builder.add_node("router", self.router_node)
        builder.add_node("rag", self.rag_node)
        builder.add_node("booking", self.booking_node)

        builder.add_edge(START, "router")

        builder.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "rag": "rag",
                "booking": "booking",
            },
        )

        builder.add_edge("rag", END)
        builder.add_edge("booking", END)

        return builder.compile()

    # -----------------------------------------------------
    # Router Node
    # -----------------------------------------------------

    def router_node(self, state: GraphState) -> GraphState:
        question = state["question"]

        response = self.client.beta.chat.completions.parse(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier for a conversational "
                        "assistant.\n\n"
                        "Classify the user's message into exactly one of "
                        "these categories:\n\n"
                        "rag = questions about company information, policies, "
                        "documents, or any information that should be answered "
                        "from uploaded documents.\n\n"
                        "booking = the user wants to schedule, book, or arrange "
                        "an interview, or is providing information for an "
                        "existing interview booking.\n\n"
                        "Return only the structured classification."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            response_format=RouterDecision,
        )

        decision = response.choices[0].message.parsed

        if decision is None:
            raise ValueError("Router could not classify the request.")

        route = decision.route.lower().strip()

        if route not in {"rag", "booking"}:
            route = "rag"

        return {
            **state,
            "route": route,
        }

    # -----------------------------------------------------
    # Decide which node to execute
    # -----------------------------------------------------

    def route_decision(self, state: GraphState) -> str:
        return state["route"]

    # -----------------------------------------------------
    # RAG Node
    # -----------------------------------------------------

    def rag_node(self, state: GraphState) -> GraphState:
        session_id = state["session_id"]
        question = state["question"]

        history = self.memory.get_history(session_id)

        query_vector = self.embedding_service.embed_text(question)

        retrieved_chunks = self.vector_store.search_similar_chunks(
            query_vector=query_vector,
            limit=5,
        )

        context_parts = []

        for chunk in retrieved_chunks:
            context_parts.append(
                f"Source: {chunk.get('filename', 'unknown')}\n"
                f"{chunk.get('text', '')}"
            )

        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful conversational RAG assistant. "
                    "Answer questions using the provided document context. "
                    "If the answer cannot be found in the context, say that "
                    "the information is not available in the provided "
                    "documents. Do not invent information."
                ),
            }
        ]

        messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Document context:\n\n{context}\n\n"
                    f"Question:\n{question}"
                ),
            }
        )

        response = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
        )

        answer = response.choices[0].message.content

        if not answer:
            answer = "I could not generate an answer."

        self.memory.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )

        self.memory.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        sources = [
            {
                "filename": chunk.get("filename"),
                "document_id": chunk.get("document_id"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
            }
            for chunk in retrieved_chunks
        ]

        return {
            **state,
            "answer": answer,
        }

    # -----------------------------------------------------
    # Booking Node
    # -----------------------------------------------------

    def booking_node(self, state: GraphState) -> GraphState:
        session_id = state["session_id"]
        question = state["question"]

        history = self.memory.get_history(session_id)

        conversation = history + [
            {
                "role": "user",
                "content": question,
            }
        ]

        conversation_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in conversation
        )

        response = self.client.beta.chat.completions.parse(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an interview booking assistant.\n\n"
                        "Extract the following information from the "
                        "conversation when available:\n"
                        "- name\n"
                        "- email\n"
                        "- interview_date\n"
                        "- interview_time\n\n"
                        "Do not invent missing information.\n"
                        "If a field is not provided, return null."
                    ),
                },
                {
                    "role": "user",
                    "content": conversation_text,
                },
            ],
            response_format=BookingDetails,
        )

        details = response.choices[0].message.parsed

        if details is None:
            raise ValueError("Could not extract booking information.")

        name = details.name or state.get("name")
        email = details.email or state.get("email")
        interview_date = (
            details.interview_date
            or state.get("interview_date")
        )
        interview_time = (
            details.interview_time
            or state.get("interview_time")
        )

        # ---------------------------------------------
        # Check missing information
        # ---------------------------------------------

        missing_fields = []

        if not name:
            missing_fields.append("name")

        if not email:
            missing_fields.append("email")

        if not interview_date:
            missing_fields.append("interview date")

        if not interview_time:
            missing_fields.append("interview time")

        # ---------------------------------------------
        # If information is missing, ask for it
        # ---------------------------------------------

        if missing_fields:
            field = missing_fields[0]

            questions = {
                "name": "Sure. May I have your name?",
                "email": "Thanks. What email address should I use?",
                "interview date": "What date would you prefer for the interview?",
                "interview time": "What time would you prefer for the interview?",
            }

            answer = questions[field]

            self.memory.add_message(
                session_id=session_id,
                role="user",
                content=question,
            )

            self.memory.add_message(
                session_id=session_id,
                role="assistant",
                content=answer,
            )

            return {
                **state,
                "name": name,
                "email": email,
                "interview_date": interview_date,
                "interview_time": interview_time,
                "answer": answer,
            }

        # ---------------------------------------------
        # All information available → save booking
        # ---------------------------------------------

        booking = save_booking(
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
        )

        answer = (
            "Your interview has been booked successfully.\n\n"
            f"Name: {name}\n"
            f"Email: {email}\n"
            f"Date: {interview_date}\n"
            f"Time: {interview_time}\n"
            f"Booking ID: {booking.id}"
        )

        self.memory.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )

        self.memory.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
        )

        return {
            **state,
            "name": name,
            "email": email,
            "interview_date": interview_date,
            "interview_time": interview_time,
            "booking_id": booking.id,
            "answer": answer,
        }

    # -----------------------------------------------------
    # Public chat method
    # -----------------------------------------------------

    def chat(
        self,
        session_id: str,
        question: str,
    ) -> dict:
        result = self.graph.invoke(
            {
                "session_id": session_id,
                "question": question,
            }
        )

        return {
            "session_id": session_id,
            "question": question,
            "answer": result.get(
                "answer",
                "I could not generate an answer.",
            ),
            "route": result.get("route"),
        }