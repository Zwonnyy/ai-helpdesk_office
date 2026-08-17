from pathlib import Path
import gc

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


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

PRETRAINED_EMBEDDING_PATH = (
    BASE_DIR
    / "models"
    / "rag_embeddings.npy"
)

V2_EMBEDDING_PATH = (
    BASE_DIR
    / "models"
    / "rag_embeddings_finetuned_v2.npy"
)

V2_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_model_v2"
)

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "answer_quality_comparison.txt"
)

DETAIL_PATH = (
    BASE_DIR
    / "reports"
    / "answer_quality_details.csv"
)


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 3

QUERY_BATCH_SIZE = 128

ENCODE_BATCH_SIZE = 64


# ============================================================
# TOP K SEARCH
# ============================================================

def retrieve_top_k(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int,
):

    all_indices = []
    all_scores = []

    total = len(query_embeddings)

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

        for scores in similarities:

            candidate_indices = (
                np.argpartition(
                    scores,
                    -top_k,
                )[-top_k:]
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

            all_indices.append(
                top_indices
            )

            all_scores.append(
                scores[
                    top_indices
                ]
            )

    return (
        np.array(all_indices),
        np.array(all_scores),
    )


# ============================================================
# ANSWER QUALITY
# ============================================================

def calculate_answer_scores(
    retrieved_indices,
    corpus_answer_embeddings,
    gold_answer_embeddings,
):

    top1_scores = []

    best3_scores = []

    mean3_scores = []

    all_scores = []

    for i in range(
        len(retrieved_indices)
    ):

        indices = (
            retrieved_indices[i]
        )

        retrieved_embeddings = (
            corpus_answer_embeddings[
                indices
            ]
        )

        gold_embedding = (
            gold_answer_embeddings[i]
        )

        # 모두 normalized embedding
        answer_scores = (
            retrieved_embeddings
            @ gold_embedding
        )

        top1_scores.append(
            float(
                answer_scores[0]
            )
        )

        best3_scores.append(
            float(
                np.max(
                    answer_scores
                )
            )
        )

        mean3_scores.append(
            float(
                np.mean(
                    answer_scores
                )
            )
        )

        all_scores.append(
            answer_scores
        )

    return (
        np.array(top1_scores),
        np.array(best3_scores),
        np.array(mean3_scores),
        np.array(all_scores),
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    name,
    top1,
    best3,
    mean3,
):

    return {
        "name": name,

        "top1_mean": float(
            np.mean(top1)
        ),

        "top1_median": float(
            np.median(top1)
        ),

        "best3_mean": float(
            np.mean(best3)
        ),

        "best3_median": float(
            np.median(best3)
        ),

        "mean3": float(
            np.mean(mean3)
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ANSWER QUALITY EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    test_df = pd.read_csv(
        TEST_PATH
    )

    corpus_df = pd.read_csv(
        CORPUS_PATH
    )

    print(
        f"Test Samples : {len(test_df):,}"
    )

    print(
        f"Corpus       : {len(corpus_df):,}"
    )

    # --------------------------------------------------------
    # 기존 Corpus Embeddings
    # --------------------------------------------------------

    pretrained_corpus_embeddings = np.load(
        PRETRAINED_EMBEDDING_PATH
    )

    v2_corpus_embeddings = np.load(
        V2_EMBEDDING_PATH
    )

    if (
        len(corpus_df)
        != len(
            pretrained_corpus_embeddings
        )
    ):
        raise ValueError(
            "Pretrained embedding 개수가 "
            "Corpus와 다릅니다."
        )

    if (
        len(corpus_df)
        != len(
            v2_corpus_embeddings
        )
    ):
        raise ValueError(
            "V2 embedding 개수가 "
            "Corpus와 다릅니다."
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
        f"Device       : {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU          :",
            torch.cuda.get_device_name(0),
        )

    # ========================================================
    # STEP 1
    # Pretrained Model
    #
    # - Pretrained Retriever Query
    # - Answer Judge Embedding
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 1 - PRETRAINED MODEL")
    print("=" * 70)

    base_model = SentenceTransformer(
        BASE_MODEL,
        device=device,
    )

    test_texts = (
        test_df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        "Pretrained Query Embedding..."
    )

    pretrained_query_embeddings = (
        base_model.encode(
            test_texts,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Answer embeddings
    #
    # 이 모델은 Retriever 평가가 아니라
    # Answer ↔ Gold Answer 의미 유사도 측정용 Judge
    # --------------------------------------------------------

    corpus_answers = (
        corpus_df["answer"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    gold_answers = (
        test_df["answer"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print()
    print(
        "Corpus Answer Embedding..."
    )

    corpus_answer_embeddings = (
        base_model.encode(
            corpus_answers,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        .astype(
            np.float32
        )
    )

    print()
    print(
        "Gold Answer Embedding..."
    )

    gold_answer_embeddings = (
        base_model.encode(
            gold_answers,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # GPU 메모리 정리
    # --------------------------------------------------------

    del base_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # STEP 2
    # V2 Query Embedding
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 2 - FINE-TUNED V2 MODEL")
    print("=" * 70)

    v2_model = SentenceTransformer(
        str(
            V2_MODEL_PATH
        ),
        device=device,
    )

    print(
        "V2 Query Embedding..."
    )

    v2_query_embeddings = (
        v2_model.encode(
            test_texts,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        .astype(
            np.float32
        )
    )

    del v2_model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # STEP 3
    # Retrieval
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 3 - RETRIEVAL")
    print("=" * 70)

    print(
        "Pretrained Search..."
    )

    (
        pretrained_indices,
        pretrained_retrieval_scores,
    ) = retrieve_top_k(
        pretrained_query_embeddings,
        pretrained_corpus_embeddings,
        TOP_K,
    )

    print(
        "V2 Search..."
    )

    (
        v2_indices,
        v2_retrieval_scores,
    ) = retrieve_top_k(
        v2_query_embeddings,
        v2_corpus_embeddings,
        TOP_K,
    )

    # ========================================================
    # STEP 4
    # Answer Quality
    # ========================================================

    print()
    print("=" * 70)
    print("STEP 4 - ANSWER QUALITY")
    print("=" * 70)

    (
        pretrained_top1,
        pretrained_best3,
        pretrained_mean3,
        pretrained_answer_scores,
    ) = calculate_answer_scores(
        pretrained_indices,
        corpus_answer_embeddings,
        gold_answer_embeddings,
    )

    (
        v2_top1,
        v2_best3,
        v2_mean3,
        v2_answer_scores,
    ) = calculate_answer_scores(
        v2_indices,
        corpus_answer_embeddings,
        gold_answer_embeddings,
    )

    pretrained_summary = summarize(
        "Pretrained",
        pretrained_top1,
        pretrained_best3,
        pretrained_mean3,
    )

    v2_summary = summarize(
        "Fine-tuned V2",
        v2_top1,
        v2_best3,
        v2_mean3,
    )

    # ========================================================
    # WIN RATE
    # ========================================================

    v2_win_rate = float(
        np.mean(
            v2_best3
            > pretrained_best3
        )
    )

    pretrained_win_rate = float(
        np.mean(
            pretrained_best3
            > v2_best3
        )
    )

    tie_rate = float(
        np.mean(
            np.isclose(
                pretrained_best3,
                v2_best3,
                atol=1e-6,
            )
        )
    )

    # ========================================================
    # REPORT
    # ========================================================

    result = f"""
ANSWER QUALITY COMPARISON
======================================================================

Judge Model:
{BASE_MODEL}

Test Samples:
{len(test_df)}

NOTE:
Answer similarity is an automatic semantic proxy.
It is not equivalent to human answer-quality evaluation.


PRETRAINED RETRIEVER
----------------------------------------------------------------------

Top1 Answer Similarity Mean:
{pretrained_summary["top1_mean"]:.4f}

Top1 Answer Similarity Median:
{pretrained_summary["top1_median"]:.4f}

Best Answer@3 Similarity Mean:
{pretrained_summary["best3_mean"]:.4f}

Best Answer@3 Similarity Median:
{pretrained_summary["best3_median"]:.4f}

Mean Answer@3 Similarity:
{pretrained_summary["mean3"]:.4f}


FINE-TUNED V2 RETRIEVER
----------------------------------------------------------------------

Top1 Answer Similarity Mean:
{v2_summary["top1_mean"]:.4f}

Top1 Answer Similarity Median:
{v2_summary["top1_median"]:.4f}

Best Answer@3 Similarity Mean:
{v2_summary["best3_mean"]:.4f}

Best Answer@3 Similarity Median:
{v2_summary["best3_median"]:.4f}

Mean Answer@3 Similarity:
{v2_summary["mean3"]:.4f}


COMPARISON
----------------------------------------------------------------------

Top1 Mean Improvement:
{v2_summary["top1_mean"] - pretrained_summary["top1_mean"]:+.4f}

Best@3 Mean Improvement:
{v2_summary["best3_mean"] - pretrained_summary["best3_mean"]:+.4f}

Mean@3 Improvement:
{v2_summary["mean3"] - pretrained_summary["mean3"]:+.4f}


Best@3 Win Rate

V2 Wins:
{v2_win_rate:.4f}

Pretrained Wins:
{pretrained_win_rate:.4f}

Tie:
{tie_rate:.4f}
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

    # ========================================================
    # DETAIL CSV
    # ========================================================

    detail_rows = []

    for i in range(
        len(test_df)
    ):

        row = test_df.iloc[i]

        pretrained_top_index = (
            pretrained_indices[
                i,
                0
            ]
        )

        v2_top_index = (
            v2_indices[
                i,
                0
            ]
        )

        pretrained_row = (
            corpus_df.iloc[
                pretrained_top_index
            ]
        )

        v2_row = (
            corpus_df.iloc[
                v2_top_index
            ]
        )

        detail_rows.append(
            {
                "test_subject": (
                    row["subject"]
                ),

                "test_type": (
                    row["type"]
                ),

                "test_queue": (
                    row["queue"]
                ),

                "gold_answer": (
                    row["answer"]
                ),

                # --------------------------------------------
                # Pretrained
                # --------------------------------------------

                "pretrained_subject": (
                    pretrained_row[
                        "subject"
                    ]
                ),

                "pretrained_answer": (
                    pretrained_row[
                        "answer"
                    ]
                ),

                "pretrained_retrieval_score": float(
                    pretrained_retrieval_scores[
                        i,
                        0
                    ]
                ),

                "pretrained_answer_similarity_top1": float(
                    pretrained_top1[i]
                ),

                "pretrained_answer_similarity_best3": float(
                    pretrained_best3[i]
                ),

                # --------------------------------------------
                # V2
                # --------------------------------------------

                "v2_subject": (
                    v2_row[
                        "subject"
                    ]
                ),

                "v2_answer": (
                    v2_row[
                        "answer"
                    ]
                ),

                "v2_retrieval_score": float(
                    v2_retrieval_scores[
                        i,
                        0
                    ]
                ),

                "v2_answer_similarity_top1": float(
                    v2_top1[i]
                ),

                "v2_answer_similarity_best3": float(
                    v2_best3[i]
                ),

                # --------------------------------------------
                # Delta
                # --------------------------------------------

                "best3_delta": float(
                    v2_best3[i]
                    - pretrained_best3[i]
                ),
            }
        )

    detail_df = pd.DataFrame(
        detail_rows
    )

    detail_df.to_csv(
        DETAIL_PATH,
        index=False,
    )

    print()
    print(
        f"Report:\n{REPORT_PATH}"
    )

    print()
    print(
        f"Details:\n{DETAIL_PATH}"
    )


if __name__ == "__main__":
    main()