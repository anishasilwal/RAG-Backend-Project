from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    document_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
    )

    file_type: Mapped[str] = mapped_column(
        String(20),
    )

    chunking_strategy: Mapped[str] = mapped_column(
        String(50),
    )

    chunk_size: Mapped[int] = mapped_column(
        Integer,
    )

    chunk_overlap: Mapped[int] = mapped_column(
        Integer,
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )


class InterviewBooking(Base):
    __tablename__ = "interview_bookings"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(150),
    )

    email: Mapped[str] = mapped_column(
        String(255),
    )

    interview_date: Mapped[str] = mapped_column(
        String(20),
    )

    interview_time: Mapped[str] = mapped_column(
        String(20),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )