from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import linear_kernel


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_test.csv"
)

CORPUS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_train.csv"
)

TFIDF_PATH = (
    BASE_DIR
    / "models"
    / "answer_retriever.joblib"
)

EMBEDDING_PATH = (
    BASE_DIR
    / "models"
    / "rag_embeddings.npy"
)

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "hybrid_retriever_evaluation.txt"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 3

# 각 Retriever에서 후보 몇 개를 가져올지
CANDIDATE_K = 30

# RRF 상수
RRF_K = 60


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    tfidf_indices,
    semantic_indices,
):
    scores = {}

    # --------------------------------------------------------
    # TF-IDF rank
    # --------------------------------------------------------

    for rank, index in enumerate(
        tfidf_indices,
        start=1,
    ):
        scores[index] = (
            scores.get(index, 0)
            + 1 / (RRF_K + rank)
        )

    # --------------------------------------------------------
    # Semantic rank
    # --------------------------------------------------------

    for rank, index in enumerate(
        semantic_indices,
        start=1,
    ):
        scores[index] = (
            scores.get(index, 0)
            + 1 / (RRF_K + rank)
        )

    sorted_indices = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return sorted_indices[:TOP_K]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HYBRID RETRIEVER EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    test_df = pd.read_csv(
        TEST_PATH
    )

    corpus_df = pd.read_csv(
        CORPUS_PATH
    )

    print(
        f"Test Samples: {len(test_df):,}"
    )

    print(
        f"Corpus: {len(corpus_df):,}"
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    tfidf_bundle = joblib.load(
        TFIDF_PATH
    )

    vectorizer = tfidf_bundle[
        "vectorizer"
    ]

    tfidf_matrix = tfidf_bundle[
        "matrix"
    ]

    print(
        f"TF-IDF Matrix: {tfidf_matrix.shape}"
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    corpus_embeddings = np.load(
        EMBEDDING_PATH
    )

    print(
        f"Embedding Matrix: "
        f"{corpus_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Semantic Model
    # --------------------------------------------------------

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    # --------------------------------------------------------
    # Test embeddings
    # --------------------------------------------------------

    test_texts = (
        test_df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print()
    print(
        "Test Embedding 생성..."
    )

    test_embeddings = model.encode(
        test_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        np.float32
    )

    # ========================================================
    # Metrics
    # ========================================================

    top1_type = 0
    top1_queue = 0
    top1_priority = 0

    topk_type = 0
    topk_queue = 0
    topk_priority = 0

    total = len(
        test_df
    )

    print()
    print(
        "Hybrid Retrieval 평가 시작..."
    )

    # ========================================================
    # Evaluation
    # ========================================================

    for i, row in test_df.iterrows():

        text = str(
            row["text"]
        )

        # ----------------------------------------------------
        # TF-IDF SEARCH
        # ----------------------------------------------------

        query_tfidf = (
            vectorizer.transform(
                [text]
            )
        )

        tfidf_scores = (
            linear_kernel(
                query_tfidf,
                tfidf_matrix,
            )
            .flatten()
        )

        tfidf_candidates = (
            np.argpartition(
                tfidf_scores,
                -CANDIDATE_K,
            )[-CANDIDATE_K:]
        )

        tfidf_indices = (
            tfidf_candidates[
                np.argsort(
                    tfidf_scores[
                        tfidf_candidates
                    ]
                )[::-1]
            ]
        )

        # ----------------------------------------------------
        # SEMANTIC SEARCH
        # ----------------------------------------------------

        query_embedding = (
            test_embeddings[i]
        )

        semantic_scores = (
            corpus_embeddings
            @ query_embedding
        )

        semantic_candidates = (
            np.argpartition(
                semantic_scores,
                -CANDIDATE_K,
            )[-CANDIDATE_K:]
        )

        semantic_indices = (
            semantic_candidates[
                np.argsort(
                    semantic_scores[
                        semantic_candidates
                    ]
                )[::-1]
            ]
        )

        # ----------------------------------------------------
        # RRF
        # ----------------------------------------------------

        hybrid_indices = (
            reciprocal_rank_fusion(
                tfidf_indices,
                semantic_indices,
            )
        )

        retrieved = (
            corpus_df.iloc[
                hybrid_indices
            ]
        )

        top1 = (
            retrieved.iloc[0]
        )

        # ====================================================
        # TOP 1
        # ====================================================

        if (
            top1["type"]
            == row["type"]
        ):
            top1_type += 1

        if (
            top1["queue"]
            == row["queue"]
        ):
            top1_queue += 1

        if (
            top1["priority"]
            == row["priority"]
        ):
            top1_priority += 1

        # ====================================================
        # TOP K
        # ====================================================

        if (
            row["type"]
            in retrieved[
                "type"
            ].values
        ):
            topk_type += 1

        if (
            row["queue"]
            in retrieved[
                "queue"
            ].values
        ):
            topk_queue += 1

        if (
            row["priority"]
            in retrieved[
                "priority"
            ].values
        ):
            topk_priority += 1

        if (
            (i + 1) % 250 == 0
        ):
            print(
                f"{i + 1:,} / "
                f"{total:,}"
            )

    # ========================================================
    # Scores
    # ========================================================

    type_1 = (
        top1_type / total
    )

    queue_1 = (
        top1_queue / total
    )

    priority_1 = (
        top1_priority / total
    )

    type_k = (
        topk_type / total
    )

    queue_k = (
        topk_queue / total
    )

    priority_k = (
        topk_priority / total
    )

    result = f"""
HYBRID RETRIEVER EVALUATION
=====================================

Method:
TF-IDF + Semantic Embedding + RRF

Candidate K:
{CANDIDATE_K}

RRF K:
{RRF_K}

Test Samples:
{total}


TOP 1

Type Match:
{type_1:.4f}

Queue Match:
{queue_1:.4f}

Priority Match:
{priority_1:.4f}


TOP {TOP_K}

Type Match@{TOP_K}:
{type_k:.4f}

Queue Match@{TOP_K}:
{queue_k:.4f}

Priority Match@{TOP_K}:
{priority_k:.4f}
"""

    print(
        result
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        result,
        encoding="utf-8",
    )

    print(
        f"Report 저장:"
        f"\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()