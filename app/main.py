from contextlib import asynccontextmanager

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
)

from sqlalchemy.orm import Session


# ============================================================
# SERVICES
# ============================================================

from app.answer_retriever import (
    AnswerRetriever,
)

from app.llm_service import (
    LLMService,
)

from app.model_service import (
    ModelService,
)


# ============================================================
# DATABASE
# ============================================================

from app.database import (
    DATABASE_PATH,
    get_db,
    init_db,
)


# ============================================================
# AUDIT
# ============================================================

from app.audit_service import (
    list_ticket_events,
)


# ============================================================
# SCHEMAS
# ============================================================

from app.schemas import (
    TicketApproveRequest,
    TicketAssistResponse,
    TicketCreateRequest,
    TicketDraftResponse,
    TicketEventResponse,
    TicketPredictionResponse,
    TicketRejectRequest,
    TicketRequest,
    TicketResponse,
    TicketSubmitRequest,
)


# ============================================================
# TICKET SERVICE
# ============================================================

from app.ticket_service import (
    InvalidTicketStateError,
    TicketNotFoundError,
    analyze_ticket,
    approve_ticket,
    create_ticket,
    get_ticket,
    list_tickets,
    reject_ticket,
    submit_for_approval,
)


# ============================================================
# SERVICE INSTANCES
# ============================================================

model_service = ModelService()

answer_retriever = AnswerRetriever()

llm_service = LLMService()


# ============================================================
# LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    print()
    print("=" * 70)
    print("AI HELPDESK STARTUP")
    print("=" * 70)

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    print()
    print("[1/3] DATABASE INITIALIZE")

    init_db()

    # --------------------------------------------------------
    # CLASSIFICATION MODELS
    #
    # TYPE
    # QUEUE
    # PRIORITY
    # --------------------------------------------------------

    print()
    print("[2/3] CLASSIFICATION MODELS LOAD")

    model_service.load_models()

    # --------------------------------------------------------
    # FINE-TUNED SEMANTIC RETRIEVER V2
    # --------------------------------------------------------

    print()
    print("[3/3] SEMANTIC RETRIEVER LOAD")

    answer_retriever.load()

    # --------------------------------------------------------
    # READY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AI HELPDESK READY")
    print("=" * 70)

    print(
        f"Database: {DATABASE_PATH}"
    )

    print(
        f"Device: {model_service.device}"
    )

    print()

    yield

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AI HELPDESK SHUTDOWN")
    print("=" * 70)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(

    title="AI Helpdesk API",

    description=(
        "IT Support Ticket Classification + "
        "Fine-tuned Semantic Retrieval + "
        "Database Workflow + "
        "Human Approval + "
        "Audit Trail"
    ),

    version="0.5.0",

    lifespan=lifespan,
)


# ============================================================
# ERROR HANDLER HELPER
# ============================================================

def raise_service_error(
    error: Exception,
):

    # --------------------------------------------------------
    # TICKET NOT FOUND
    # --------------------------------------------------------

    if isinstance(
        error,
        TicketNotFoundError,
    ):

        raise HTTPException(
            status_code=404,
            detail=str(
                error
            ),
        )

    # --------------------------------------------------------
    # INVALID WORKFLOW STATE
    # --------------------------------------------------------

    if isinstance(
        error,
        InvalidTicketStateError,
    ):

        raise HTTPException(
            status_code=409,
            detail=str(
                error
            ),
        )

    # --------------------------------------------------------
    # UNKNOWN ERROR
    # --------------------------------------------------------

    raise error


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "service": (
            "AI Helpdesk"
        ),

        "version": (
            "0.5.0"
        ),

        "status": (
            "running"
        ),

        "features": [
            "ticket-classification",
            "semantic-retrieval",
            "human-approval",
            "audit-trail",
        ],
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "device": str(
            model_service.device
        ),

        "models": list(
            model_service
            .models
            .keys()
        ),

        "retriever": (
            answer_retriever.loaded
        ),

        "database": str(
            DATABASE_PATH
        ),

        "llm_model": (
            llm_service.model_name
        ),
    }


# ============================================================
# ============================================================
#
# LEGACY / AI DIRECT API
#
# ============================================================
# ============================================================


# ============================================================
# PREDICT
#
# Ticket을 DB에 저장하지 않고
# TYPE / QUEUE / PRIORITY만 즉시 분석
# ============================================================

@app.post(
    "/predict",
    response_model=(
        TicketPredictionResponse
    ),
)
def predict(
    request: TicketRequest,
):

    predictions = (
        model_service.predict(

            subject=request.subject,

            body=request.body,
        )
    )

    return predictions


# ============================================================
# ASSIST
#
# Classification
# +
# Fine-tuned Semantic Retrieval
# ============================================================

@app.post(
    "/assist",
    response_model=(
        TicketAssistResponse
    ),
)
def assist(

    request: TicketRequest,

    top_k: int = Query(
        default=3,
        ge=1,
        le=10,
    ),
):

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    predictions = (
        model_service.predict(

            subject=request.subject,

            body=request.body,
        )
    )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    similar_tickets = (
        answer_retriever.search(

            subject=request.subject,

            body=request.body,

            top_k=top_k,
        )
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "type": (
            predictions[
                "type"
            ]
        ),

        "queue": (
            predictions[
                "queue"
            ]
        ),

        "priority": (
            predictions[
                "priority"
            ]
        ),

        "similar_tickets": (
            similar_tickets
        ),
    }


# ============================================================
# DRAFT
#
# Classification
# +
# Retrieval
# +
# Local LLM
#
# 현재 Ollama CUDA 문제가 있을 경우
# API 전체를 죽이지 않고 503 반환
# ============================================================

@app.post(
    "/draft",
    response_model=(
        TicketDraftResponse
    ),
)
def draft(

    request: TicketRequest,

    top_k: int = Query(
        default=3,
        ge=1,
        le=5,
    ),
):

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    predictions = (
        model_service.predict(

            subject=request.subject,

            body=request.body,
        )
    )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    similar_tickets = (
        answer_retriever.search(

            subject=request.subject,

            body=request.body,

            top_k=top_k,
        )
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    try:

        draft_answer = (
            llm_service.generate_answer(

                subject=request.subject,

                body=request.body,

                predictions=predictions,

                similar_tickets=(
                    similar_tickets
                ),
            )
        )

    except Exception as error:

        raise HTTPException(

            status_code=503,

            detail=(
                "Local LLM is currently unavailable. "
                "Classification and retrieval are "
                "working normally. "
                f"Error: {error}"
            ),
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "type": (
            predictions[
                "type"
            ]
        ),

        "queue": (
            predictions[
                "queue"
            ]
        ),

        "priority": (
            predictions[
                "priority"
            ]
        ),

        "similar_tickets": (
            similar_tickets
        ),

        "draft_answer": (
            draft_answer
        ),
    }


# ============================================================
# ============================================================
#
# TICKET WORKFLOW API
#
# ============================================================
# ============================================================


# ============================================================
# CREATE TICKET
#
# NONE
#  ↓
# PENDING
#
# Audit:
# TICKET_CREATED
# ============================================================

@app.post(
    "/tickets",
    response_model=(
        TicketResponse
    ),
    status_code=201,
)
def create_ticket_api(

    request: TicketCreateRequest,

    db: Session = Depends(
        get_db
    ),
):

    try:

        ticket = create_ticket(

            db=db,

            subject=request.subject,

            body=request.body,
        )

        return ticket

    except Exception as error:

        raise_service_error(
            error
        )


# ============================================================
# LIST TICKETS
# ============================================================

@app.get(
    "/tickets",
    response_model=list[
        TicketResponse
    ],
)
def list_tickets_api(

    offset: int = Query(
        default=0,
        ge=0,
    ),

    limit: int = Query(
        default=50,
        ge=1,
        le=100,
    ),

    db: Session = Depends(
        get_db
    ),
):

    try:

        tickets = list_tickets(

            db=db,

            offset=offset,

            limit=limit,
        )

        return tickets

    except Exception as error:

        raise_service_error(
            error
        )


# ============================================================
# GET TICKET
# ============================================================

@app.get(
    "/tickets/{ticket_id}",
    response_model=(
        TicketResponse
    ),
)
def get_ticket_api(

    ticket_id: int,

    db: Session = Depends(
        get_db
    ),
):

    try:

        ticket = get_ticket(

            db=db,

            ticket_id=ticket_id,
        )

        return ticket

    except Exception as error:

        raise_service_error(
            error
        )


# ============================================================
# ANALYZE TICKET
#
# PENDING
#   ↓
# AI CLASSIFICATION
#   +
# V2 SEMANTIC RETRIEVAL
#   ↓
# ANALYZED
#
# Audit:
# AI_ANALYZED
# ============================================================

@app.post(
    "/tickets/{ticket_id}/analyze",
    response_model=(
        TicketResponse
    ),
)
def analyze_ticket_api(

    ticket_id: int,

    top_k: int = Query(
        default=3,
        ge=1,
        le=10,
    ),

    db: Session = Depends(
        get_db
    ),
):

    try:

        # ----------------------------------------------------
        # GET TICKET
        # ----------------------------------------------------

        ticket = get_ticket(

            db=db,

            ticket_id=ticket_id,
        )

        # ----------------------------------------------------
        # ANALYZE
        # ----------------------------------------------------

        ticket = analyze_ticket(

            db=db,

            ticket=ticket,

            model_service=(
                model_service
            ),

            answer_retriever=(
                answer_retriever
            ),

            top_k=top_k,
        )

        return ticket

    except Exception as error:

        raise_service_error(
            error
        )


# ============================================================
# SUBMIT FOR APPROVAL
#
# ANALYZED
#    ↓
# WAITING_APPROVAL
#
# 또는
#
# REJECTED
#    ↓
# WAITING_APPROVAL
#
# Audit:
# SUBMITTED_FOR_APPROVAL
# ============================================================

@app.post(
    "/tickets/{ticket_id}/submit-for-approval",
    response_model=(
        TicketResponse
    ),
)
def submit_ticket_api(

    ticket_id: int,

    request: TicketSubmitRequest,

    db: Session = Depends(
        get_db
    ),
):

    try:

        ticket = get_ticket(

            db=db,

            ticket_id=ticket_id,
        )

        ticket = submit_for_approval(

            db=db,

            ticket=ticket,

            draft_answer=(
                request.draft_answer
            ),
        )

        return ticket

    except Exception as error:

        raise_service_error(
            error
        )


# ============================================================
# APPROVE
#
# WAITING_APPROVAL
#        ↓
#     APPROVED
#
# Audit:
# APPROVED
# ============================================================

@app.post(
    "/tickets/{ticket_id}/approve",
    response_model=(
        TicketResponse
    ),
)
def approve_ticket_api(

    ticket_id: int,

    request: TicketApproveRequest,

    db: Session = Depends(
        get_db
    ),
):

    try:

        ticket = get_ticket(

            db=db,

            ticket_id=ticket_id,
        )

        ticket = approve_ticket(

            db=db,

            ticket=ticket,

            final_answer=(
                request.final_answer
            ),
        )

        return ticket

    except Exception as error:

        raise_service_error(
            error
        )


# ============================================================
# REJECT
#
# WAITING_APPROVAL
#        ↓
#     REJECTED
#
# Audit:
# REJECTED
# ============================================================

@app.post(
    "/tickets/{ticket_id}/reject",
    response_model=(
        TicketResponse
    ),
)
def reject_ticket_api(

    ticket_id: int,

    request: TicketRejectRequest,

    db: Session = Depends(
        get_db
    ),
):

    try:

        ticket = get_ticket(

            db=db,

            ticket_id=ticket_id,
        )

        ticket = reject_ticket(

            db=db,

            ticket=ticket,

            reason=(
                request.reason
            ),
        )

        return ticket

    except Exception as error:

        raise_service_error(
            error
        )


# ============================================================
# ============================================================
#
# AUDIT TRAIL API
#
# ============================================================
# ============================================================


# ============================================================
# GET TICKET EVENTS
#
# Ticket의 전체 상태 변경 이력 조회
# ============================================================

@app.get(
    "/tickets/{ticket_id}/events",
    response_model=list[
        TicketEventResponse
    ],
)
def get_ticket_events_api(

    ticket_id: int,

    db: Session = Depends(
        get_db
    ),
):

    try:

        # ----------------------------------------------------
        # Ticket 존재 여부 확인
        # ----------------------------------------------------

        get_ticket(

            db=db,

            ticket_id=ticket_id,
        )

        # ----------------------------------------------------
        # Event List
        # ----------------------------------------------------

        events = list_ticket_events(

            db=db,

            ticket_id=ticket_id,
        )

        return events

    except Exception as error:

        raise_service_error(
            error
        )