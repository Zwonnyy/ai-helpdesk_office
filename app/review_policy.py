from dataclasses import dataclass


# ============================================================
# CLASSIFICATION THRESHOLDS
#
# 현재 classifier threshold는
# 운영 정책 기반 초기 기준값.
#
# Retrieval threshold는
# V3 validation calibration 결과 기반.
# ============================================================

TYPE_CONFIDENCE_MIN = 0.70

QUEUE_CONFIDENCE_MIN = 0.75

PRIORITY_CONFIDENCE_MIN = 0.65


# ============================================================
# RETRIEVAL THRESHOLD
#
# V3 Dev Calibration
#
# Threshold = 0.75
#
# Accepted Precision:
# 0.9515
#
# Coverage:
# 0.7509
#
# Review Rate:
# 0.2491
#
# Failure Capture Rate:
# 0.7759
# ============================================================

RETRIEVAL_SIMILARITY_MIN = 0.75


# ============================================================
# CRITICAL THRESHOLDS
# ============================================================

TYPE_CONFIDENCE_CRITICAL = 0.50

QUEUE_CONFIDENCE_CRITICAL = 0.55

PRIORITY_CONFIDENCE_CRITICAL = 0.45


# ============================================================
# RETRIEVAL CRITICAL THRESHOLD
#
# Failed Retrieval Similarity
# 25th Percentile
#
# Calibration Result:
# 0.6243
# ============================================================

RETRIEVAL_SIMILARITY_CRITICAL = 0.6243


# ============================================================
# RISK LEVEL
# ============================================================

RISK_LOW = "LOW"

RISK_MEDIUM = "MEDIUM"

RISK_HIGH = "HIGH"


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class ReviewRiskResult:

    risk_level: str

    review_required: bool

    reasons: list[str]


# ============================================================
# EVALUATE
# ============================================================

def evaluate_review_risk(
    type_confidence: float,
    queue_confidence: float,
    priority_confidence: float,
    retrieval_similarity: float,
) -> ReviewRiskResult:

    reasons = []

    critical = False

    # ========================================================
    # TYPE
    # ========================================================

    if (
        type_confidence
        < TYPE_CONFIDENCE_MIN
    ):

        reasons.append(
            "LOW_TYPE_CONFIDENCE"
        )

    if (
        type_confidence
        < TYPE_CONFIDENCE_CRITICAL
    ):

        critical = True

    # ========================================================
    # QUEUE
    # ========================================================

    if (
        queue_confidence
        < QUEUE_CONFIDENCE_MIN
    ):

        reasons.append(
            "LOW_QUEUE_CONFIDENCE"
        )

    if (
        queue_confidence
        < QUEUE_CONFIDENCE_CRITICAL
    ):

        critical = True

    # ========================================================
    # PRIORITY
    # ========================================================

    if (
        priority_confidence
        < PRIORITY_CONFIDENCE_MIN
    ):

        reasons.append(
            "LOW_PRIORITY_CONFIDENCE"
        )

    if (
        priority_confidence
        < PRIORITY_CONFIDENCE_CRITICAL
    ):

        critical = True

    # ========================================================
    # RETRIEVAL
    # ========================================================

    if (
        retrieval_similarity
        < RETRIEVAL_SIMILARITY_MIN
    ):

        reasons.append(
            "LOW_RETRIEVAL_SIMILARITY"
        )

    if (
        retrieval_similarity
        < RETRIEVAL_SIMILARITY_CRITICAL
    ):

        critical = True

    # ========================================================
    # RISK LEVEL
    # ========================================================

    if (
        critical
        or len(reasons) >= 2
    ):

        risk_level = (
            RISK_HIGH
        )

    elif len(reasons) == 1:

        risk_level = (
            RISK_MEDIUM
        )

    else:

        risk_level = (
            RISK_LOW
        )

    # ========================================================
    # HUMAN REVIEW
    #
    # LOW:
    # 별도의 강화 Review 불필요
    #
    # MEDIUM/HIGH:
    # 강화 Human Review 필요
    #
    # 현재 v1.0에서는 LOW도 최종 승인 자체는
    # 사람이 수행함.
    # ========================================================

    review_required = (
        risk_level
        != RISK_LOW
    )

    return ReviewRiskResult(

        risk_level=risk_level,

        review_required=review_required,

        reasons=reasons,
    )