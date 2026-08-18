from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import (
    SentenceTransformer,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DEV_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "v3_dev.csv"
)

CORPUS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_train.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_model_v3"
)

EMBEDDING_PATH = (
    BASE_DIR
    / "models"
    / "rag_embeddings_finetuned_v3.npy"
)

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "v3_retrieval_threshold_calibration.csv"
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 64

SEARCH_BATCH_SIZE = 128

THRESHOLDS = np.arange(
    0.50,
    0.951,
    0.01,
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V3 RETRIEVAL THRESHOLD CALIBRATION")
    print("=" * 70)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    dev_df = pd.read_csv(
        DEV_PATH
    )

    corpus_df = pd.read_csv(
        CORPUS_PATH
    )

    corpus_embeddings = np.load(
        EMBEDDING_PATH
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Dev Samples : {len(dev_df):,}"
    )

    print(
        f"Corpus      : {len(corpus_df):,}"
    )

    print(
        f"Device      : {device}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = SentenceTransformer(
        str(
            MODEL_PATH
        ),
        device=device,
    )

    # --------------------------------------------------------
    # QUERY EMBEDDINGS
    # --------------------------------------------------------

    texts = (
        dev_df[
            "text"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print()
    print(
        "Query Embeddings 생성..."
    )

    query_embeddings = (
        model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # TOP 1 SEARCH
    # --------------------------------------------------------

    top1_indices = []

    top1_scores = []

    for start in range(
        0,
        len(
            query_embeddings
        ),
        SEARCH_BATCH_SIZE,
    ):

        end = min(
            start
            + SEARCH_BATCH_SIZE,
            len(
                query_embeddings
            ),
        )

        similarities = (
            query_embeddings[
                start:end
            ]
            @ corpus_embeddings.T
        )

        indices = np.argmax(
            similarities,
            axis=1,
        )

        scores = similarities[
            np.arange(
                len(
                    indices
                )
            ),
            indices,
        ]

        top1_indices.extend(
            indices.tolist()
        )

        top1_scores.extend(
            scores.tolist()
        )

    top1_indices = np.asarray(
        top1_indices
    )

    top1_scores = np.asarray(
        top1_scores
    )

    # ========================================================
    # SUCCESS DEFINITION
    #
    # Retriever의 TOP1 Ticket이
    # Type + Queue 모두 일치하면 성공으로 정의
    #
    # Priority는 retrieval 자체보다 별도 classifier 성격이
    # 강하기 때문에 여기서는 제외
    # ========================================================

    success = []

    for i in range(
        len(dev_df)
    ):

        query_row = dev_df.iloc[
            i
        ]

        retrieved_row = (
            corpus_df.iloc[
                top1_indices[
                    i
                ]
            ]
        )

        type_match = (
            query_row[
                "type"
            ]
            == retrieved_row[
                "type"
            ]
        )

        queue_match = (
            query_row[
                "queue"
            ]
            == retrieved_row[
                "queue"
            ]
        )

        success.append(
            type_match
            and queue_match
        )

    success = np.asarray(
        success,
        dtype=bool,
    )

    # ========================================================
    # BASIC DISTRIBUTION
    # ========================================================

    successful_scores = (
        top1_scores[
            success
        ]
    )

    failed_scores = (
        top1_scores[
            ~success
        ]
    )

    print()
    print("=" * 70)
    print("SIMILARITY DISTRIBUTION")
    print("=" * 70)

    print(
        f"Overall Mean : "
        f"{np.mean(top1_scores):.4f}"
    )

    print(
        f"Success Mean : "
        f"{np.mean(successful_scores):.4f}"
    )

    print(
        f"Failure Mean : "
        f"{np.mean(failed_scores):.4f}"
    )

    print(
        f"Success Rate : "
        f"{np.mean(success):.4f}"
    )

    # ========================================================
    # THRESHOLD SWEEP
    # ========================================================

    results = []

    for threshold in THRESHOLDS:

        accepted = (
            top1_scores
            >= threshold
        )

        reviewed = (
            ~accepted
        )

        accepted_count = int(
            accepted.sum()
        )

        reviewed_count = int(
            reviewed.sum()
        )

        # ----------------------------------------------------
        # Accepted precision
        #
        # review 없이 통과시킨 Ticket 중
        # 실제 성공률
        # ----------------------------------------------------

        if accepted_count > 0:

            accepted_precision = float(
                success[
                    accepted
                ].mean()
            )

        else:

            accepted_precision = (
                np.nan
            )

        # ----------------------------------------------------
        # Coverage
        # ----------------------------------------------------

        coverage = (
            accepted_count
            / len(
                dev_df
            )
        )

        review_rate = (
            reviewed_count
            / len(
                dev_df
            )
        )

        # ----------------------------------------------------
        # Failed retrieval 중
        # 얼마나 review 대상으로 잡혔나
        # ----------------------------------------------------

        total_failures = int(
            (
                ~success
            ).sum()
        )

        if total_failures > 0:

            failure_capture_rate = float(
                reviewed[
                    ~success
                ].mean()
            )

        else:

            failure_capture_rate = (
                0.0
            )

        results.append(
            {
                "threshold": (
                    round(
                        float(
                            threshold
                        ),
                        2,
                    )
                ),

                "coverage": (
                    coverage
                ),

                "review_rate": (
                    review_rate
                ),

                "accepted_precision": (
                    accepted_precision
                ),

                "failure_capture_rate": (
                    failure_capture_rate
                ),

                "accepted_count": (
                    accepted_count
                ),

                "reviewed_count": (
                    reviewed_count
                ),
            }
        )

    result_df = pd.DataFrame(
        results
    )

    # ========================================================
    # RECOMMENDED THRESHOLD
    #
    # 95% 이상의 precision을 만족하면서
    # 가장 많은 Ticket을 자동 통과시키는 최소 threshold
    # ========================================================

    safe_candidates = (
        result_df[
            result_df[
                "accepted_precision"
            ]
            >= 0.95
        ]
    )

    if not safe_candidates.empty:

        recommended = (
            safe_candidates.iloc[
                0
            ]
        )

    else:

        recommended = None

    # ========================================================
    # CRITICAL THRESHOLD
    #
    # 실패 Ticket similarity의 25 percentile 아래를
    # 매우 낮은 retrieval confidence로 간주
    # ========================================================

    if len(
        failed_scores
    ) > 0:

        critical_threshold = float(
            np.quantile(
                failed_scores,
                0.25,
            )
        )

    else:

        critical_threshold = 0.50

    # ========================================================
    # OUTPUT
    # ========================================================

    result_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("CALIBRATION RESULT")
    print("=" * 70)

    if recommended is not None:

        print(
            "Recommended Normal Threshold:"
        )

        print(
            f"{recommended['threshold']:.2f}"
        )

        print()

        print(
            "Accepted Precision:"
        )

        print(
            f"{recommended['accepted_precision']:.4f}"
        )

        print()

        print(
            "Coverage:"
        )

        print(
            f"{recommended['coverage']:.4f}"
        )

        print()

        print(
            "Review Rate:"
        )

        print(
            f"{recommended['review_rate']:.4f}"
        )

        print()

        print(
            "Failure Capture Rate:"
        )

        print(
            f"{recommended['failure_capture_rate']:.4f}"
        )

    else:

        print(
            "95% precision을 만족하는 "
            "threshold가 없습니다."
        )

    print()
    print(
        "Recommended Critical Threshold:"
    )

    print(
        f"{critical_threshold:.4f}"
    )

    print()
    print(
        f"Report:\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()