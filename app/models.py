from datetime import (
    datetime,
    timezone,
)

from enum import Enum

from sqlalchemy import (
    DateTime,
    Float,
    JSON,
    String,
    Text,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from app.database import Base


# ============================================================
# TIME
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )


# ============================================================
# STATUS
# ============================================================

class TicketStatus(
    str,
    Enum,
):

    PENDING = "PENDING"

    ANALYZED = "ANALYZED"

    WAITING_APPROVAL = (
        "WAITING_APPROVAL"
    )

    APPROVED = "APPROVED"

    REJECTED = "REJECTED"


# ============================================================
# TICKET
# ============================================================

class Ticket(
    Base
):

    __tablename__ = "tickets"

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    # --------------------------------------------------------
    # Original Ticket
    # --------------------------------------------------------

    subject: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    body: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # --------------------------------------------------------
    # Workflow Status
    # --------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=TicketStatus.PENDING.value,
        index=True,
    )

    # --------------------------------------------------------
    # AI Prediction
    # --------------------------------------------------------

    predicted_type: Mapped[
        str | None
    ] = mapped_column(
        String(100),
        nullable=True,
    )

    type_confidence: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    predicted_queue: Mapped[
        str | None
    ] = mapped_column(
        String(200),
        nullable=True,
    )

    queue_confidence: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    predicted_priority: Mapped[
        str | None
    ] = mapped_column(
        String(100),
        nullable=True,
    )

    priority_confidence: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    # --------------------------------------------------------
    # Retrieval Result
    #
    # TOP 3 Ticket 정보를 JSON으로 저장
    # --------------------------------------------------------

    similar_tickets: Mapped[
        list | None
    ] = mapped_column(
        JSON,
        nullable=True,
    )

    # --------------------------------------------------------
    # LLM / Human Approval
    # --------------------------------------------------------

    draft_answer: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    final_answer: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    review_comment: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    created_at: Mapped[
        datetime
    ] = mapped_column(
        DateTime(
            timezone=True
        ),
        default=utc_now,
        nullable=False,
    )

    updated_at: Mapped[
        datetime
    ] = mapped_column(
        DateTime(
            timezone=True
        ),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    analyzed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(
            timezone=True
        ),
        nullable=True,
    )

    reviewed_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(
            timezone=True
        ),
        nullable=True,
    )