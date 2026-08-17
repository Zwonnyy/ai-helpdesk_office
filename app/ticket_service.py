from datetime import (
    datetime,
    timezone,
)

from sqlalchemy import (
    select,
)

from sqlalchemy.orm import (
    Session,
)

from app.models import (
    Ticket,
    TicketStatus,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class TicketNotFoundError(
    Exception
):
    pass


class InvalidTicketStateError(
    Exception
):
    pass


# ============================================================
# TIME
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )


# ============================================================
# CREATE
# ============================================================

def create_ticket(
    db: Session,
    subject: str,
    body: str,
) -> Ticket:

    ticket = Ticket(
        subject=subject.strip(),
        body=body.strip(),
        status=(
            TicketStatus
            .PENDING
            .value
        ),
    )

    db.add(
        ticket
    )

    db.commit()

    db.refresh(
        ticket
    )

    return ticket


# ============================================================
# GET
# ============================================================

def get_ticket(
    db: Session,
    ticket_id: int,
) -> Ticket:

    ticket = db.get(
        Ticket,
        ticket_id,
    )

    if ticket is None:

        raise TicketNotFoundError(
            f"Ticket {ticket_id}를 "
            f"찾을 수 없습니다."
        )

    return ticket


# ============================================================
# LIST
# ============================================================

def list_tickets(
    db: Session,
    offset: int = 0,
    limit: int = 50,
):

    statement = (
        select(
            Ticket
        )
        .order_by(
            Ticket.id.desc()
        )
        .offset(
            offset
        )
        .limit(
            limit
        )
    )

    return list(
        db.scalars(
            statement
        ).all()
    )


# ============================================================
# ANALYZE
# ============================================================

def analyze_ticket(
    db: Session,
    ticket: Ticket,
    model_service,
    answer_retriever,
    top_k: int = 3,
) -> Ticket:

    allowed_status = {
        TicketStatus.PENDING.value,
        TicketStatus.ANALYZED.value,
        TicketStatus.REJECTED.value,
    }

    if ticket.status not in allowed_status:

        raise InvalidTicketStateError(
            "현재 상태에서는 "
            "AI 분석을 수행할 수 없습니다. "
            f"status={ticket.status}"
        )

    # --------------------------------------------------------
    # STEP 1
    # Classification
    # --------------------------------------------------------

    predictions = (
        model_service.predict(
            subject=ticket.subject,
            body=ticket.body,
        )
    )

    # --------------------------------------------------------
    # STEP 2
    # Fine-tuned V2 Retrieval
    # --------------------------------------------------------

    similar_tickets = (
        answer_retriever.search(
            subject=ticket.subject,
            body=ticket.body,
            top_k=top_k,
        )
    )

    # --------------------------------------------------------
    # DB
    # --------------------------------------------------------

    ticket.predicted_type = (
        predictions[
            "type"
        ][
            "label"
        ]
    )

    ticket.type_confidence = float(
        predictions[
            "type"
        ][
            "confidence"
        ]
    )

    ticket.predicted_queue = (
        predictions[
            "queue"
        ][
            "label"
        ]
    )

    ticket.queue_confidence = float(
        predictions[
            "queue"
        ][
            "confidence"
        ]
    )

    ticket.predicted_priority = (
        predictions[
            "priority"
        ][
            "label"
        ]
    )

    ticket.priority_confidence = float(
        predictions[
            "priority"
        ][
            "confidence"
        ]
    )

    ticket.similar_tickets = (
        similar_tickets
    )

    ticket.status = (
        TicketStatus
        .ANALYZED
        .value
    )

    ticket.analyzed_at = (
        utc_now()
    )

    # 재분석한 경우
    # 기존 reviewer 상태 초기화
    ticket.reviewed_at = None

    db.commit()

    db.refresh(
        ticket
    )

    return ticket


# ============================================================
# SUBMIT FOR APPROVAL
# ============================================================

def submit_for_approval(
    db: Session,
    ticket: Ticket,
    draft_answer: str | None = None,
) -> Ticket:

    allowed_status = {
        TicketStatus.ANALYZED.value,
        TicketStatus.REJECTED.value,
    }

    if ticket.status not in allowed_status:

        raise InvalidTicketStateError(
            "승인 요청은 ANALYZED 또는 "
            "REJECTED 상태에서만 가능합니다. "
            f"status={ticket.status}"
        )

    if draft_answer is not None:

        draft_answer = (
            draft_answer.strip()
        )

        if draft_answer:

            ticket.draft_answer = (
                draft_answer
            )

    ticket.status = (
        TicketStatus
        .WAITING_APPROVAL
        .value
    )

    ticket.review_comment = None

    ticket.reviewed_at = None

    db.commit()

    db.refresh(
        ticket
    )

    return ticket


# ============================================================
# APPROVE
# ============================================================

def approve_ticket(
    db: Session,
    ticket: Ticket,
    final_answer: str | None = None,
) -> Ticket:

    if (
        ticket.status
        != TicketStatus
        .WAITING_APPROVAL
        .value
    ):

        raise InvalidTicketStateError(
            "WAITING_APPROVAL 상태의 "
            "Ticket만 승인할 수 있습니다. "
            f"status={ticket.status}"
        )

    if final_answer is not None:

        final_answer = (
            final_answer.strip()
        )

        if final_answer:

            ticket.final_answer = (
                final_answer
            )

    elif ticket.draft_answer:

        # 수정하지 않고 승인하면
        # draft를 final로 승격
        ticket.final_answer = (
            ticket.draft_answer
        )

    ticket.status = (
        TicketStatus
        .APPROVED
        .value
    )

    ticket.reviewed_at = (
        utc_now()
    )

    ticket.review_comment = None

    db.commit()

    db.refresh(
        ticket
    )

    return ticket


# ============================================================
# REJECT
# ============================================================

def reject_ticket(
    db: Session,
    ticket: Ticket,
    reason: str,
) -> Ticket:

    if (
        ticket.status
        != TicketStatus
        .WAITING_APPROVAL
        .value
    ):

        raise InvalidTicketStateError(
            "WAITING_APPROVAL 상태의 "
            "Ticket만 반려할 수 있습니다. "
            f"status={ticket.status}"
        )

    ticket.status = (
        TicketStatus
        .REJECTED
        .value
    )

    ticket.review_comment = (
        reason.strip()
    )

    ticket.reviewed_at = (
        utc_now()
    )

    db.commit()

    db.refresh(
        ticket
    )

    return ticket