from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


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

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_model_v2"
)

EMBEDDING_PATH = (
    BASE_DIR
    / "models"
    / "rag_embeddings_finetuned_v2.npy"
)

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "finetuned_retriever_v2_evaluation.txt"
)

TOP_K = 3
QUERY_BATCH_SIZE = 128


def main():

    print("=" * 70)
    print("FINE-TUNED V2 RETRIEVER EVALUATION")
    print("=" * 70)

    test_df = pd.read_csv(
        TEST_PATH
    )

    corpus_df = pd.read_csv(
        CORPUS_PATH
    )

    corpus_embeddings = np.load(
        EMBEDDING_PATH
    )

    print(
        f"Test Samples: {len(test_df):,}"
    )

    print(
        f"Corpus: {len(corpus_df):,}"
    )

    print(
        f"Embeddings: {corpus_embeddings.shape}"
    )

    if len(corpus_df) != len(corpus_embeddings):
        raise ValueError(
            "Corpus와 Embedding 개수가 다릅니다."
        )

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

    print()
    print(
        "Fine-tuned V2 Model 로드..."
    )

    model = SentenceTransformer(
        str(MODEL_PATH),
        device=device,
    )

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

    query_embeddings = model.encode(
        test_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        np.float32
    )

    top1_type = 0
    top1_queue = 0
    top1_priority = 0

    topk_type = 0
    topk_queue = 0
    topk_priority = 0

    similarity_scores = []

    total = len(
        test_df
    )

    print()
    print(
        "V2 Semantic Search 평가 시작..."
    )

    for start in range(
        0,
        total,
        QUERY_BATCH_SIZE,
    ):

        end = min(
            start + QUERY_BATCH_SIZE,
            total,
        )

        query_batch = (
            query_embeddings[
                start:end
            ]
        )

        similarities = (
            query_batch
            @ corpus_embeddings.T
        )

        for local_index in range(
            end - start
        ):

            global_index = (
                start
                + local_index
            )

            scores = (
                similarities[
                    local_index
                ]
            )

            candidate_indices = (
                np.argpartition(
                    scores,
                    -TOP_K,
                )[-TOP_K:]
            )

            top_indices = (
                candidate_indices[
                    np.argsort(
                        scores[
                            candidate_indices
                        ]
                    )[::-1]
                ]
            )

            query_row = (
                test_df.iloc[
                    global_index
                ]
            )

            retrieved = (
                corpus_df.iloc[
                    top_indices
                ]
            )

            top1 = (
                retrieved.iloc[0]
            )

            similarity_scores.append(
                float(
                    scores[
                        top_indices[0]
                    ]
                )
            )

            # -------------------------
            # TOP 1
            # -------------------------

            if (
                top1["type"]
                == query_row["type"]
            ):
                top1_type += 1

            if (
                top1["queue"]
                == query_row["queue"]
            ):
                top1_queue += 1

            if (
                top1["priority"]
                == query_row["priority"]
            ):
                top1_priority += 1

            # -------------------------
            # TOP 3
            # -------------------------

            if (
                query_row["type"]
                in retrieved[
                    "type"
                ].values
            ):
                topk_type += 1

            if (
                query_row["queue"]
                in retrieved[
                    "queue"
                ].values
            ):
                topk_queue += 1

            if (
                query_row["priority"]
                in retrieved[
                    "priority"
                ].values
            ):
                topk_priority += 1

        print(
            f"{end:,} / {total:,}"
        )

    type_1 = (
        top1_type / total
    )

    queue_1 = (
        top1_queue / total
    )

    priority_1 = (
        top1_priority / total
    )

    type_3 = (
        topk_type / total
    )

    queue_3 = (
        topk_queue / total
    )

    priority_3 = (
        topk_priority / total
    )

    mean_similarity = float(
        np.mean(
            similarity_scores
        )
    )

    result = f"""
FINE-TUNED V2 RETRIEVER EVALUATION
=====================================

Model:
Helpdesk Fine-tuned Semantic Model V2

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
{type_3:.4f}

Queue Match@{TOP_K}:
{queue_3:.4f}

Priority Match@{TOP_K}:
{priority_3:.4f}


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

    print(
        f"Report 저장:\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()