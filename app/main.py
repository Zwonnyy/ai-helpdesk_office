from contextlib import (
    asynccontextmanager,
)

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
)

from sqlalchemy.orm import Session


from app.answer_retriever import (
    AnswerRetriever,
)

from app.database import (
    DATABASE_PATH,
    get_db,
    init_db,
)

from app.llm_service import (
    LLMService,
)

from app.model_service import (
    ModelService,
)

from app.schemas import (
    TicketApproveRequest,
    TicketAssistResponse,
    TicketCreateRequest,
    TicketDraftResponse,
    TicketPredictionResponse,
    TicketRejectRequest,
    TicketRequest,
    TicketResponse,
    TicketSubmitRequest,
)

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
# SERVICES
# ============================================================

model_service = (
    ModelService()
)

answer_retriever = (
    AnswerRetriever()
)

llm_service = (
    LLMService()
)


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
    # Database
    # --------------------------------------------------------

    init_db()

    # --------------------------------------------------------
    # Classification Models
    # --------------------------------------------------------

    model_service.load_models()

    # --------------------------------------------------------
    # Fine-tuned V2 Retriever
    # --------------------------------------------------------

    answer_retriever.load()

    print()
    print("=" * 70)
    print("AI HELPDESK READY")
    print("=" * 70)

    yield

    print()
    print(
        "AI Helpdesk API 종료"
    )


# ============================================================
# APP
# ============================================================

app = FastAPI(

    title=(
        "AI Helpdesk API"
    ),

    description=(
        "Ticket Classification + "
        "Fine-tuned Semantic Retrieval + "
        "Database Workflow + "
        "Human Approval"
    ),

    version="0.4.0",

    lifespan=lifespan,
)


# ============================================================
# ERROR HELPER
# ============================================================

def raise_service_error(
    error: Exception,
):

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
            "0.4.0"
        ),

        "status": (
            "running"
        ),
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
    }


# ============================================================
# OLD API
# PREDICTION
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

    return model_service.predict(

        subject=request.subject,

        body=request.body,
    )


# ============================================================
# OLD API
# ASSIST
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

    predictions = (
        model_service.predict(
            subject=request.subject,
            body=request.body,
        )
    )

    similar_tickets = (
        answer_retriever.search(
            subject=request.subject,
            body=request.body,
            top_k=top_k,
        )
    )

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
# OLD API
# DRAFT
#
# Ollama가 현재 CUDA 문제라
# 실패하면 503으로만 처리
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

    predictions = (
        model_service.predict(
            subject=request.subject,
            body=request.body,
        )
    )

    similar_tickets = (
        answer_retriever.search(
            subject=request.subject,
            body=request.body,
            top_k=top_k,
        )
    )

    try:

        draft_answer = (
            llm_service
            .generate_answer(

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
                "Local LLM is currently "
                "unavailable. "
                "Classification and retrieval "
                "are working normally. "
                f"Error: {error}"
            ),
        )

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
# DB API
# CREATE TICKET
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

    return create_ticket(

        db=db,

        subject=request.subject,

        body=request.body,
    )


# ============================================================
# DB API
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

    return list_tickets(

        db=db,

        offset=offset,

        limit=limit,
    )


# ============================================================
# DB API
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

        return get_ticket(
            db=db,
            ticket_id=ticket_id,
        )

    except Exception as error:

        raise_service_error(
            error
        )


# ============================================================
# DB API
# ANALYZE
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

        ticket = get_ticket(
            db=db,
            ticket_id=ticket_id,
        )

        return analyze_ticket(

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

    except Exception as error:

        raise_service_error(
            error
        )


# ============================================================
# DB API
# SUBMIT FOR APPROVAL
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

        return submit_for_approval(

            db=db,

            ticket=ticket,

            draft_answer=(
                request.draft_answer
            ),
        )

    except Exception as error:

        raise_service_error(
            error
        )


# ============================================================
# DB API
# APPROVE
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

        return approve_ticket(

            db=db,

            ticket=ticket,

            final_answer=(
                request.final_answer
            ),
        )

    except Exception as error:

        raise_service_error(
            error
        )


# ============================================================
# DB API
# REJECT
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

        return reject_ticket(

            db=db,

            ticket=ticket,

            reason=(
                request.reason
            ),
        )

    except Exception as error:

        raise_service_error(
            error
        )