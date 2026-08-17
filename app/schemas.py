from datetime import datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.models import (
    TicketStatus,
)


# ============================================================
# BASIC TICKET REQUEST
# ============================================================

class TicketRequest(
    BaseModel
):

    subject: str = Field(
        min_length=1,
        examples=[
            "VPN connection unavailable"
        ],
    )

    body: str = Field(
        min_length=1,
        examples=[
            "All employees cannot "
            "connect to the VPN."
        ],
    )


# ============================================================
# AI PREDICTION
# ============================================================

class PredictionResult(
    BaseModel
):

    label: str

    confidence: float


class TicketPredictionResponse(
    BaseModel
):

    type: PredictionResult

    queue: PredictionResult

    priority: PredictionResult


# ============================================================
# RETRIEVAL
# ============================================================

class SimilarTicket(
    BaseModel
):

    score: float

    subject: str

    body: str

    answer: str

    type: str

    queue: str

    priority: str

    language: str


class TicketAssistResponse(
    BaseModel
):

    type: PredictionResult

    queue: PredictionResult

    priority: PredictionResult

    similar_tickets: list[
        SimilarTicket
    ]


# ============================================================
# LLM
# ============================================================

class TicketDraftResponse(
    BaseModel
):

    type: PredictionResult

    queue: PredictionResult

    priority: PredictionResult

    similar_tickets: list[
        SimilarTicket
    ]

    draft_answer: str


# ============================================================
# DB CREATE
# ============================================================

class TicketCreateRequest(
    BaseModel
):

    subject: str = Field(
        min_length=1,
    )

    body: str = Field(
        min_length=1,
    )


# ============================================================
# APPROVAL
# ============================================================

class TicketSubmitRequest(
    BaseModel
):

    draft_answer: (
        str | None
    ) = None


class TicketApproveRequest(
    BaseModel
):

    final_answer: (
        str | None
    ) = None


class TicketRejectRequest(
    BaseModel
):

    reason: str = Field(
        min_length=1,
    )


# ============================================================
# DB RESPONSE
# ============================================================

class TicketResponse(
    BaseModel
):

    model_config = ConfigDict(
        from_attributes=True
    )

    id: int

    subject: str

    body: str

    status: TicketStatus

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    predicted_type: (
        str | None
    ) = None

    type_confidence: (
        float | None
    ) = None

    predicted_queue: (
        str | None
    ) = None

    queue_confidence: (
        float | None
    ) = None

    predicted_priority: (
        str | None
    ) = None

    priority_confidence: (
        float | None
    ) = None

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    similar_tickets: (
        list[SimilarTicket]
        | None
    ) = None

    # --------------------------------------------------------
    # Human Approval
    # --------------------------------------------------------

    draft_answer: (
        str | None
    ) = None

    final_answer: (
        str | None
    ) = None

    review_comment: (
        str | None
    ) = None

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    created_at: datetime

    updated_at: datetime

    analyzed_at: (
        datetime | None
    ) = None

    reviewed_at: (
        datetime | None
    ) = None