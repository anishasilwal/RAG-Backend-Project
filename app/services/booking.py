from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import InterviewBooking


def save_booking(
    name: str,
    email: str,
    interview_date: str,
    interview_time: str,
) -> InterviewBooking:
    db: Session = SessionLocal()

    try:
        booking = InterviewBooking(
            name=name,
            email=email,
            interview_date=interview_date,
            interview_time=interview_time,
        )

        db.add(booking)
        db.commit()
        db.refresh(booking)

        return booking

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()