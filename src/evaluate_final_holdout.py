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

HOLDOUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_holdout.csv"
)

CORPUS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_train.csv"
)

V2_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_model_v2"
)

V3_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_model_v3"
)

V2_EMBEDDING_PATH = (
    BASE_DIR
    / "models"
    / "rag_embeddings_finetuned_v2.npy"
)

V3_EMBEDDING_PATH = (
    BASE_DIR
    / "models"
    / "rag_embeddings_finetuned_v3.npy"
)

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "final_holdout_evaluation.txt"
)

DETAIL_PATH = (
    BASE_DIR
    / "reports"
    / "final_holdout_details.csv"
)


# ============================================================
# CONFIG
# ============================================================

JUDGE_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 3

SEARCH_BATCH_SIZE = 128

ENCODE_BATCH_SIZE = 64

EPSILON = 1e-4


# ============================================================
# QUERY ENCODE
# ============================================================

def encode_queries(
    model_path,
    texts,
    device,
):

    model = SentenceTransformer(
        str(model_path),
        device=device,
    )

    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        np.float32
    )

    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return embeddings


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    query_embeddings,
    corpus_embeddings,
):

    all_indices = []

    all_scores = []

    total = len(
        query_embeddings
    )

    for start in range(
        0,
        total,
        SEARCH_BATCH_SIZE,
    ):

        end = min(
            start
            + SEARCH_BATCH_SIZE,
            total,
        )

        similarities = (
            query_embeddings[
                start:end
            ]
            @ corpus_embeddings.T
        )

        for scores in similarities:

            candidates = np.argpartition(
                scores,
                -TOP_K,
            )[-TOP_K:]

            top_indices = candidates[
                np.argsort(
                    scores[
                        candidates
                    ]
                )[::-1]
            ]

            all_indices.append(
                top_indices
            )

            all_scores.append(
                scores[
                    top_indices
                ]
            )

    return (
        np.asarray(
            all_indices
        ),
        np.asarray(
            all_scores
        ),
    )


# ============================================================
# LABEL METRICS
# ============================================================

def label_metrics(
    query_df,
    corpus_df,
    indices,
):

    metrics = {}

    for column in [
        "type",
        "queue",
        "priority",
    ]:

        top1_hits = 0

        top3_hits = 0

        for i in range(
            len(query_df)
        ):

            target = (
                query_df.iloc[i][
                    column
                ]
            )

            retrieved = (
                corpus_df.iloc[
                    indices[i]
                ][
                    column
                ]
                .values
            )

            if (
                retrieved[0]
                == target
            ):

                top1_hits += 1

            if (
                target
                in retrieved
            ):

                top3_hits += 1

        metrics[
            f"{column}_at_1"
        ] = (
            top1_hits
            / len(query_df)
        )

        metrics[
            f"{column}_at_3"
        ] = (
            top3_hits
            / len(query_df)
        )

    return metrics


# ============================================================
# ANSWER QUALITY
# ============================================================

def answer_quality(
    indices,
    corpus_answer_embeddings,
    gold_answer_embeddings,
):

    top1_scores = []

    best3_scores = []

    mean3_scores = []

    for i in range(
        len(indices)
    ):

        retrieved_embeddings = (
            corpus_answer_embeddings[
                indices[i]
            ]
        )

        gold_embedding = (
            gold_answer_embeddings[
                i
            ]
        )

        similarities = (
            retrieved_embeddings
            @ gold_embedding
        )

        top1_scores.append(
            float(
                similarities[0]
            )
        )

        best3_scores.append(
            float(
                np.max(
                    similarities
                )
            )
        )

        mean3_scores.append(
            float(
                np.mean(
                    similarities
                )
            )
        )

    return (
        np.asarray(
            top1_scores
        ),
        np.asarray(
            best3_scores
        ),
        np.asarray(
            mean3_scores
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL HOLDOUT EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    holdout_df = pd.read_csv(
        HOLDOUT_PATH
    )

    corpus_df = pd.read_csv(
        CORPUS_PATH
    )

    v2_corpus_embeddings = np.load(
        V2_EMBEDDING_PATH
    )

    v3_corpus_embeddings = np.load(
        V3_EMBEDDING_PATH
    )

    print(
        f"Holdout Samples : {len(holdout_df):,}"
    )

    print(
        f"Corpus          : {len(corpus_df):,}"
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
        f"Device          : {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU             :",
            torch.cuda.get_device_name(
                0
            ),
        )

    texts = (
        holdout_df[
            "text"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # ========================================================
    # V2 QUERY EMBEDDINGS
    # ========================================================

    print()
    print("=" * 70)
    print("V2 QUERY EMBEDDINGS")
    print("=" * 70)

    v2_queries = encode_queries(
        V2_MODEL_PATH,
        texts,
        device,
    )

    # ========================================================
    # V3 QUERY EMBEDDINGS
    # ========================================================

    print()
    print("=" * 70)
    print("V3 QUERY EMBEDDINGS")
    print("=" * 70)

    v3_queries = encode_queries(
        V3_MODEL_PATH,
        texts,
        device,
    )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    print()
    print("=" * 70)
    print("RETRIEVAL")
    print("=" * 70)

    print(
        "V2 Search..."
    )

    (
        v2_indices,
        v2_scores,
    ) = retrieve(
        v2_queries,
        v2_corpus_embeddings,
    )

    print(
        "V3 Search..."
    )

    (
        v3_indices,
        v3_scores,
    ) = retrieve(
        v3_queries,
        v3_corpus_embeddings,
    )

    # ========================================================
    # LABEL PROXY METRICS
    # ========================================================

    v2_metrics = label_metrics(
        holdout_df,
        corpus_df,
        v2_indices,
    )

    v3_metrics = label_metrics(
        holdout_df,
        corpus_df,
        v3_indices,
    )

    # ========================================================
    # ANSWER RELEVANCE JUDGE
    # ========================================================

    print()
    print("=" * 70)
    print("ANSWER RELEVANCE")
    print("=" * 70)

    judge_model = SentenceTransformer(
        JUDGE_MODEL,
        device=device,
    )

    corpus_answers = (
        corpus_df[
            "answer"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    gold_answers = (
        holdout_df[
            "answer"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        "Corpus Answer Embeddings..."
    )

    corpus_answer_embeddings = (
        judge_model.encode(
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

    print(
        "Gold Answer Embeddings..."
    )

    gold_answer_embeddings = (
        judge_model.encode(
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

    # ========================================================
    # ANSWER QUALITY
    # ========================================================

    (
        v2_answer_top1,
        v2_answer_best3,
        v2_answer_mean3,
    ) = answer_quality(
        v2_indices,
        corpus_answer_embeddings,
        gold_answer_embeddings,
    )

    (
        v3_answer_top1,
        v3_answer_best3,
        v3_answer_mean3,
    ) = answer_quality(
        v3_indices,
        corpus_answer_embeddings,
        gold_answer_embeddings,
    )

    # ========================================================
    # ANSWER WIN RATE
    # ========================================================

    answer_delta = (
        v3_answer_best3
        - v2_answer_best3
    )

    v3_wins = int(
        (
            answer_delta
            > EPSILON
        ).sum()
    )

    v2_wins = int(
        (
            answer_delta
            < -EPSILON
        ).sum()
    )

    ties = int(
        (
            np.abs(
                answer_delta
            )
            <= EPSILON
        ).sum()
    )

    non_ties = (
        v3_wins
        + v2_wins
    )

    if non_ties:

        v3_non_tie_win_rate = (
            v3_wins
            / non_ties
        )

    else:

        v3_non_tie_win_rate = (
            0.0
        )

    # ========================================================
    # SAME TOP1 RATE
    # ========================================================

    same_top1_rate = float(
        np.mean(
            v2_indices[:, 0]
            == v3_indices[:, 0]
        )
    )

    # ========================================================
    # REPORT
    # ========================================================

    result = f"""
FINAL HOLDOUT EVALUATION
======================================================================

Samples:
{len(holdout_df)}


V2
----------------------------------------------------------------------

Type@1:
{v2_metrics["type_at_1"]:.4f}

Queue@1:
{v2_metrics["queue_at_1"]:.4f}

Priority@1:
{v2_metrics["priority_at_1"]:.4f}

Type@3:
{v2_metrics["type_at_3"]:.4f}

Queue@3:
{v2_metrics["queue_at_3"]:.4f}

Priority@3:
{v2_metrics["priority_at_3"]:.4f}

Mean Top1 Retrieval Similarity:
{np.mean(v2_scores[:, 0]):.4f}

Answer Top1 Mean:
{np.mean(v2_answer_top1):.4f}

Answer Best@3 Mean:
{np.mean(v2_answer_best3):.4f}

Answer Mean@3:
{np.mean(v2_answer_mean3):.4f}


V3
----------------------------------------------------------------------

Type@1:
{v3_metrics["type_at_1"]:.4f}

Queue@1:
{v3_metrics["queue_at_1"]:.4f}

Priority@1:
{v3_metrics["priority_at_1"]:.4f}

Type@3:
{v3_metrics["type_at_3"]:.4f}

Queue@3:
{v3_metrics["queue_at_3"]:.4f}

Priority@3:
{v3_metrics["priority_at_3"]:.4f}

Mean Top1 Retrieval Similarity:
{np.mean(v3_scores[:, 0]):.4f}

Answer Top1 Mean:
{np.mean(v3_answer_top1):.4f}

Answer Best@3 Mean:
{np.mean(v3_answer_best3):.4f}

Answer Mean@3:
{np.mean(v3_answer_mean3):.4f}


V3 - V2
----------------------------------------------------------------------

Type@1 Delta:
{v3_metrics["type_at_1"] - v2_metrics["type_at_1"]:+.4f}

Queue@1 Delta:
{v3_metrics["queue_at_1"] - v2_metrics["queue_at_1"]:+.4f}

Priority@1 Delta:
{v3_metrics["priority_at_1"] - v2_metrics["priority_at_1"]:+.4f}

Type@3 Delta:
{v3_metrics["type_at_3"] - v2_metrics["type_at_3"]:+.4f}

Queue@3 Delta:
{v3_metrics["queue_at_3"] - v2_metrics["queue_at_3"]:+.4f}

Priority@3 Delta:
{v3_metrics["priority_at_3"] - v2_metrics["priority_at_3"]:+.4f}

Answer Top1 Delta:
{np.mean(v3_answer_top1) - np.mean(v2_answer_top1):+.6f}

Answer Best@3 Delta:
{np.mean(v3_answer_best3) - np.mean(v2_answer_best3):+.6f}

Answer Mean@3 Delta:
{np.mean(v3_answer_mean3) - np.mean(v2_answer_mean3):+.6f}


ANSWER BEST@3 COMPARISON
----------------------------------------------------------------------

V3 Wins:
{v3_wins}

V2 Wins:
{v2_wins}

Ties:
{ties}

V3 Non-Tie Win Rate:
{v3_non_tie_win_rate:.4f}


RETRIEVAL CHANGE
----------------------------------------------------------------------

Same Top1 Ticket Rate:
{same_top1_rate:.4f}
"""

    print(
        result
    )

    REPORT_PATH.write_text(
        result,
        encoding="utf-8",
    )

    # ========================================================
    # DETAILS
    # ========================================================

    detail_df = pd.DataFrame(
        {
            "subject": (
                holdout_df[
                    "subject"
                ]
            ),

            "type": (
                holdout_df[
                    "type"
                ]
            ),

            "queue": (
                holdout_df[
                    "queue"
                ]
            ),

            "priority": (
                holdout_df[
                    "priority"
                ]
            ),

            "v2_top1_index": (
                v2_indices[
                    :,
                    0
                ]
            ),

            "v3_top1_index": (
                v3_indices[
                    :,
                    0
                ]
            ),

            "same_top1": (
                v2_indices[
                    :,
                    0
                ]
                == v3_indices[
                    :,
                    0
                ]
            ),

            "v2_top1_similarity": (
                v2_scores[
                    :,
                    0
                ]
            ),

            "v3_top1_similarity": (
                v3_scores[
                    :,
                    0
                ]
            ),

            "v2_answer_top1": (
                v2_answer_top1
            ),

            "v3_answer_top1": (
                v3_answer_top1
            ),

            "v2_answer_best3": (
                v2_answer_best3
            ),

            "v3_answer_best3": (
                v3_answer_best3
            ),

            "answer_best3_delta": (
                answer_delta
            ),
        }
    )

    detail_df.to_csv(
        DETAIL_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL REPORT SAVED")
    print("=" * 70)

    print(
        f"Report:\n{REPORT_PATH}"
    )

    print()

    print(
        f"Details:\n{DETAIL_PATH}"
    )


if __name__ == "__main__":
    main()