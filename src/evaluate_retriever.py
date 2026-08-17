from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import linear_kernel


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
    / "answer_corpus.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "answer_retriever.joblib"
)

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "retriever_evaluation.txt"
)

TOP_K = 3


def main():

    print("=" * 70)
    print("RETRIEVER EVALUATION")
    print("=" * 70)

    test_df = pd.read_csv(
        TEST_PATH
    )

    corpus_df = pd.read_csv(
        CORPUS_PATH
    )

    bundle = joblib.load(
        MODEL_PATH
    )

    vectorizer = bundle[
        "vectorizer"
    ]

    matrix = bundle[
        "matrix"
    ]

    type_hits = 0
    queue_hits = 0
    priority_hits = 0

    top1_type = 0
    top1_queue = 0
    top1_priority = 0

    similarity_scores = []

    total = len(
        test_df
    )

    for i, row in test_df.iterrows():

        query_vector = (
            vectorizer.transform(
                [row["text"]]
            )
        )

        similarities = (
            linear_kernel(
                query_vector,
                matrix,
            )
            .flatten()
        )

        top_indices = (
            np.argsort(
                similarities
            )[::-1][:TOP_K]
        )

        retrieved = corpus_df.iloc[
            top_indices
        ]

        top1 = retrieved.iloc[
            0
        ]

        similarity_scores.append(
            similarities[
                top_indices[0]
            ]
        )

        # TOP 1
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

        # TOP K
        if (
            row["type"]
            in retrieved[
                "type"
            ].values
        ):
            type_hits += 1

        if (
            row["queue"]
            in retrieved[
                "queue"
            ].values
        ):
            queue_hits += 1

        if (
            row["priority"]
            in retrieved[
                "priority"
            ].values
        ):
            priority_hits += 1

        if (
            (i + 1) % 500 == 0
        ):
            print(
                f"{i + 1:,} / "
                f"{total:,}"
            )

    top1_type_score = (
        top1_type / total
    )

    top1_queue_score = (
        top1_queue / total
    )

    top1_priority_score = (
        top1_priority / total
    )

    type_hit_k = (
        type_hits / total
    )

    queue_hit_k = (
        queue_hits / total
    )

    priority_hit_k = (
        priority_hits / total
    )

    mean_similarity = float(
        np.mean(
            similarity_scores
        )
    )

    result = f"""
RETRIEVER EVALUATION
=====================================

Test Samples:
{total}

TOP 1

Type Match:
{top1_type_score:.4f}

Queue Match:
{top1_queue_score:.4f}

Priority Match:
{top1_priority_score:.4f}


TOP {TOP_K}

Type Match@{TOP_K}:
{type_hit_k:.4f}

Queue Match@{TOP_K}:
{queue_hit_k:.4f}

Priority Match@{TOP_K}:
{priority_hit_k:.4f}


Mean Top1 Similarity:
{mean_similarity:.4f}
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


if __name__ == "__main__":
    main()