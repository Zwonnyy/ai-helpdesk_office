from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)

from enum import Enum

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    JSON,
    String,
    Text,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
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
# TICKET STATUS
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
# EVENT TYPE
# ============================================================

class TicketEventType(
    str,
    Enum,
):

    TICKET_CREATED = (
        "TICKET_CREATED"
    )

    AI_ANALYZED = (
        "AI_ANALYZED"
    )

    RISK_EVALUATED = (
        "RISK_EVALUATED"
    )

    SUBMITTED_FOR_APPROVAL = (
        "SUBMITTED_FOR_APPROVAL"
    )

    APPROVED = (
        "APPROVED"
    )

    REJECTED = (
        "REJECTED"
    )


# ============================================================
# TICKET
# ============================================================

class Ticket(
    Base
):

    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    # --------------------------------------------------------
    # ORIGINAL TICKET
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
    # WORKFLOW
    # --------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=TicketStatus.PENDING.value,
        index=True,
    )

    # --------------------------------------------------------
    # CLASSIFICATION
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
    # RETRIEVAL
    # --------------------------------------------------------

    similar_tickets: Mapped[
        list | None
    ] = mapped_column(
        JSON,
        nullable=True,
    )

    retrieval_top1_similarity: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    # --------------------------------------------------------
    # RISK GATE
    # --------------------------------------------------------

    risk_level: Mapped[
        str | None
    ] = mapped_column(
        String(32),
        nullable=True,
        index=True,
    )

    review_required: Mapped[
        bool
    ] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    risk_reasons: Mapped[
        list | None
    ] = mapped_column(
        JSON,
        nullable=True,
    )

    # --------------------------------------------------------
    # HUMAN APPROVAL
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
    # TIME
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

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    events: Mapped[
        list["TicketEvent"]
    ] = relationship(
        back_populates="ticket",
        cascade="all, delete-orphan",
    )


# ============================================================
# TICKET EVENT
# ============================================================

class TicketEvent(
    Base
):

    __tablename__ = (
        "ticket_events"
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    ticket_id: Mapped[int] = mapped_column(
        ForeignKey(
            "tickets.id"
        ),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    from_status: Mapped[
        str | None
    ] = mapped_column(
        String(32),
        nullable=True,
    )

    to_status: Mapped[
        str | None
    ] = mapped_column(
        String(32),
        nullable=True,
    )

    message: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    event_data: Mapped[
        dict | None
    ] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[
        datetime
    ] = mapped_column(
        DateTime(
            timezone=True
        ),
        default=utc_now,
        nullable=False,
        index=True,
    )

    ticket: Mapped[
        "Ticket"
    ] = relationship(
        back_populates="events"
    )