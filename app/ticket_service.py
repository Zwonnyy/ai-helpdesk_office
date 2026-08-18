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

from app.audit_service import (
    add_ticket_event,
)

from app.models import (
    Ticket,
    TicketEventType,
    TicketStatus,
)

from app.review_policy import (
    evaluate_review_risk,
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

    # commit 전에 ID만 먼저 확보
    db.flush()

    add_ticket_event(

        db=db,

        ticket_id=ticket.id,

        event_type=(
            TicketEventType
            .TICKET_CREATED
            .value
        ),

        from_status=None,

        to_status=(
            TicketStatus
            .PENDING
            .value
        ),

        message=(
            "Ticket created"
        ),
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

        TicketStatus
        .PENDING
        .value,

        TicketStatus
        .ANALYZED
        .value,

        TicketStatus
        .REJECTED
        .value,
    }

    if (
        ticket.status
        not in allowed_status
    ):

        raise InvalidTicketStateError(
            "현재 상태에서는 "
            "AI 분석을 수행할 수 없습니다. "
            f"status={ticket.status}"
        )

    old_status = (
        ticket.status
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    predictions = (
        model_service.predict(

            subject=ticket.subject,

            body=ticket.body,
        )
    )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    similar_tickets = (
        answer_retriever.search(

            subject=ticket.subject,

            body=ticket.body,

            top_k=top_k,
        )
    )

    retrieval_top1_similarity = float(
        similar_tickets[0]['score']
        if similar_tickets
        else 0.0
    )

    # --------------------------------------------------------
    # UPDATE
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

    ticket.retrieval_top1_similarity = (
        retrieval_top1_similarity
    )

    risk_result = evaluate_review_risk(
        type_confidence=ticket.type_confidence,
        queue_confidence=ticket.queue_confidence,
        priority_confidence=ticket.priority_confidence,
        retrieval_similarity=(
            ticket.retrieval_top1_similarity
        ),
    )

    ticket.risk_level = risk_result.risk_level
    ticket.review_required = (
        risk_result.review_required
    )
    ticket.risk_reasons = list(
        risk_result.reasons
    )

    ticket.status = (
        TicketStatus
        .ANALYZED
        .value
    )

    ticket.analyzed_at = (
        utc_now()
    )

    ticket.reviewed_at = None

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    add_ticket_event(

        db=db,

        ticket_id=ticket.id,

        event_type=(
            TicketEventType
            .AI_ANALYZED
            .value
        ),

        from_status=old_status,

        to_status=(
            TicketStatus
            .ANALYZED
            .value
        ),

        message=(
            "AI classification and "
            "retrieval completed"
        ),

        event_data={
            "type": {
                "label": (
                    ticket.predicted_type
                ),
                "confidence": (
                    ticket.type_confidence
                ),
            },

            "queue": {
                "label": (
                    ticket.predicted_queue
                ),
                "confidence": (
                    ticket.queue_confidence
                ),
            },

            "priority": {
                "label": (
                    ticket.predicted_priority
                ),
                "confidence": (
                    ticket.priority_confidence
                ),
            },

            "retrieved_count": (
                len(
                    similar_tickets
                )
            ),
        },
    )

    add_ticket_event(
        db=db,
        ticket_id=ticket.id,
        event_type=TicketEventType.RISK_EVALUATED.value,
        from_status=TicketStatus.ANALYZED.value,
        to_status=TicketStatus.ANALYZED.value,
        message='AI review risk evaluated',
        event_data={
            'risk_level': ticket.risk_level,
            'review_required': ticket.review_required,
            'reasons': ticket.risk_reasons,
            'threshold_inputs': {
                'type_confidence': ticket.type_confidence,
                'queue_confidence': ticket.queue_confidence,
                'priority_confidence': ticket.priority_confidence,
                'retrieval_similarity': ticket.retrieval_top1_similarity,
            },
        },
    )

    db.commit()

    db.refresh(
        ticket
    )

    return ticket


# ============================================================
# SUBMIT
# ============================================================

def submit_for_approval(
    db: Session,
    ticket: Ticket,
    draft_answer: str | None = None,
) -> Ticket:

    allowed_status = {

        TicketStatus
        .ANALYZED
        .value,

        TicketStatus
        .REJECTED
        .value,
    }

    if (
        ticket.status
        not in allowed_status
    ):

        raise InvalidTicketStateError(
            "승인 요청은 ANALYZED 또는 "
            "REJECTED 상태에서만 가능합니다. "
            f"status={ticket.status}"
        )

    old_status = (
        ticket.status
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

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    add_ticket_event(

        db=db,

        ticket_id=ticket.id,

        event_type=(
            TicketEventType
            .SUBMITTED_FOR_APPROVAL
            .value
        ),

        from_status=old_status,

        to_status=(
            TicketStatus
            .WAITING_APPROVAL
            .value
        ),

        message=(
            "Ticket submitted "
            "for human approval"
        ),

        event_data={
            "has_draft_answer": (
                bool(
                    ticket.draft_answer
                )
            ),
        },
    )

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

    old_status = (
        ticket.status
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

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    add_ticket_event(

        db=db,

        ticket_id=ticket.id,

        event_type=(
            TicketEventType
            .APPROVED
            .value
        ),

        from_status=old_status,

        to_status=(
            TicketStatus
            .APPROVED
            .value
        ),

        message=(
            "Ticket approved "
            "by human reviewer"
        ),

        event_data={
            "final_answer_present": (
                bool(
                    ticket.final_answer
                )
            ),
        },
    )

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

    old_status = (
        ticket.status
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

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    add_ticket_event(

        db=db,

        ticket_id=ticket.id,

        event_type=(
            TicketEventType
            .REJECTED
            .value
        ),

        from_status=old_status,

        to_status=(
            TicketStatus
            .REJECTED
            .value
        ),

        message=(
            "Ticket rejected "
            "by human reviewer"
        ),

        event_data={
            "reason": (
                ticket.review_comment
            ),
        },
    )

    db.commit()

    db.refresh(
        ticket
    )

    return ticket
