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

EMBEDDING_PATH = (
    BASE_DIR
    / "models"
    / "rag_embeddings.npy"
)

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "embedding_retriever_evaluation.txt"
)


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 3

QUERY_BATCH_SIZE = 128


def main():

    print("=" * 70)
    print("SEMANTIC RETRIEVER EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 데이터 로드
    # --------------------------------------------------------

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
    # 모델
    # --------------------------------------------------------

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    # --------------------------------------------------------
    # Test embedding 생성
    # --------------------------------------------------------

    print()
    print(
        "Test Embedding 생성..."
    )

    test_texts = (
        test_df["text"]
        .fillna("")
        .astype(str)
        .tolist()
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

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Batch Retrieval
    # --------------------------------------------------------

    print()
    print(
        "Semantic Search 평가 시작..."
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

        # normalize_embeddings=True 이므로
        # dot product == cosine similarity
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

            scores = similarities[
                local_index
            ]

            # 상위 K개 index
            candidate_indices = (
                np.argpartition(
                    scores,
                    -TOP_K,
                )[-TOP_K:]
            )

            # 실제 similarity 순 정렬
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

            # =================================================
            # TOP 1
            # =================================================

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

            # =================================================
            # TOP K
            # =================================================

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

    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    top1_type_score = (
        top1_type / total
    )

    top1_queue_score = (
        top1_queue / total
    )

    top1_priority_score = (
        top1_priority / total
    )

    topk_type_score = (
        topk_type / total
    )

    topk_queue_score = (
        topk_queue / total
    )

    topk_priority_score = (
        topk_priority / total
    )

    mean_similarity = float(
        np.mean(
            similarity_scores
        )
    )

    result = f"""
SEMANTIC RETRIEVER EVALUATION
=====================================

Model:
{MODEL_NAME}

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
{topk_type_score:.4f}

Queue Match@{TOP_K}:
{topk_queue_score:.4f}

Priority Match@{TOP_K}:
{topk_priority_score:.4f}


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
        f"Report 저장:"
        f"\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()