from sqlalchemy import (
    select,
)

from sqlalchemy.orm import (
    Session,
)

from app.models import (
    TicketEvent,
)


# ============================================================
# CREATE EVENT
# ============================================================

def add_ticket_event(
    db: Session,
    ticket_id: int,
    event_type: str,
    from_status: str | None,
    to_status: str | None,
    message: str | None = None,
    event_data: dict | None = None,
):

    event = TicketEvent(

        ticket_id=ticket_id,

        event_type=event_type,

        from_status=from_status,

        to_status=to_status,

        message=message,

        event_data=event_data,
    )

    db.add(
        event
    )

    return event


# ============================================================
# LIST EVENTS
# ============================================================

def list_ticket_events(
    db: Session,
    ticket_id: int,
):

    statement = (
        select(
            TicketEvent
        )
        .where(
            TicketEvent.ticket_id
            == ticket_id
        )
        .order_by(
            TicketEvent.id.asc()
        )
    )

    return list(
        db.scalars(
            statement
        ).all()
    )