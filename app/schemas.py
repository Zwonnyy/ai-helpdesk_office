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
# BASIC REQUEST
# ============================================================

class TicketRequest(
    BaseModel
):

    subject: str = Field(
        min_length=1
    )

    body: str = Field(
        min_length=1
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

    kb_index: int | None = None

    score: float

    subject: str

    body: str

    answer: str

    type: str

    queue: str

    priority: str

    language: str

    subject_ko: (
        str | None
    ) = None

    body_ko: (
        str | None
    ) = None

    answer_ko: (
        str | None
    ) = None

    translation_status: (
        str | None
    ) = None


class SimilarTicketTranslationResponse(BaseModel):

    kb_index: int | None
    target_language: str
    subject: str
    body: str
    answer: str
    cached: bool
    translated: bool = True
    error: str | None = None


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
# LLM DRAFT
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
# CREATE
# ============================================================

class TicketCreateRequest(
    BaseModel
):

    subject: str = Field(
        min_length=1
    )

    body: str = Field(
        min_length=1
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
        min_length=1
    )


# ============================================================
# TICKET RESPONSE
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
    # CLASSIFICATION
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
    # RETRIEVAL
    # --------------------------------------------------------

    similar_tickets: (
        list[SimilarTicket]
        | None
    ) = None

    retrieval_top1_similarity: (
        float | None
    ) = None

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk_level: (
        str | None
    ) = None

    review_required: bool = False

    risk_reasons: (
        list[str]
        | None
    ) = None

    # --------------------------------------------------------
    # HUMAN APPROVAL
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
    # TIME
    # --------------------------------------------------------

    created_at: datetime

    updated_at: datetime

    analyzed_at: (
        datetime | None
    ) = None

    reviewed_at: (
        datetime | None
    ) = None


# ============================================================
# EVENT
# ============================================================

class TicketEventResponse(
    BaseModel
):

    model_config = ConfigDict(
        from_attributes=True
    )

    id: int

    ticket_id: int

    event_type: str

    from_status: (
        str | None
    )

    to_status: (
        str | None
    )

    message: (
        str | None
    )

    event_data: (
        dict | None
    )

    created_at: datetime
