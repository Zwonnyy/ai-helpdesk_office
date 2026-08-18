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
    / "v2_vs_v3_dev_evaluation.txt"
)

DETAIL_PATH = (
    BASE_DIR
    / "reports"
    / "v2_vs_v3_dev_details.csv"
)


# Answer similarity judge
JUDGE_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 3
SEARCH_BATCH_SIZE = 128
ENCODE_BATCH_SIZE = 64


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
# RETRIEVAL
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
            start + SEARCH_BATCH_SIZE,
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

    result = {}

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

            if retrieved[0] == target:
                top1_hits += 1

            if target in retrieved:
                top3_hits += 1

        result[
            f"{column}_at_1"
        ] = (
            top1_hits
            / len(query_df)
        )

        result[
            f"{column}_at_3"
        ] = (
            top3_hits
            / len(query_df)
        )

    return result


# ============================================================
# ANSWER QUALITY
# ============================================================

def answer_quality(
    indices,
    corpus_answer_embeddings,
    gold_answer_embeddings,
):

    top1 = []
    best3 = []
    mean3 = []

    for i in range(
        len(indices)
    ):

        retrieved = (
            corpus_answer_embeddings[
                indices[i]
            ]
        )

        gold = (
            gold_answer_embeddings[i]
        )

        scores = (
            retrieved
            @ gold
        )

        top1.append(
            float(
                scores[0]
            )
        )

        best3.append(
            float(
                np.max(
                    scores
                )
            )
        )

        mean3.append(
            float(
                np.mean(
                    scores
                )
            )
        )

    return (
        np.asarray(top1),
        np.asarray(best3),
        np.asarray(mean3),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V2 VS V3 DEV EVALUATION")
    print("=" * 70)

    dev_df = pd.read_csv(
        DEV_PATH
    )

    corpus_df = pd.read_csv(
        CORPUS_PATH
    )

    v2_corpus = np.load(
        V2_EMBEDDING_PATH
    )

    v3_corpus = np.load(
        V3_EMBEDDING_PATH
    )

    print(
        f"Dev Samples : {len(dev_df):,}"
    )

    print(
        f"Corpus      : {len(corpus_df):,}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device      : {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU         :",
            torch.cuda.get_device_name(0),
        )

    texts = (
        dev_df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # ========================================================
    # V2
    # ========================================================

    print()
    print("=" * 70)
    print("V2 QUERY EMBEDDING")
    print("=" * 70)

    v2_queries = encode_queries(
        V2_MODEL_PATH,
        texts,
        device,
    )

    # ========================================================
    # V3
    # ========================================================

    print()
    print("=" * 70)
    print("V3 QUERY EMBEDDING")
    print("=" * 70)

    v3_queries = encode_queries(
        V3_MODEL_PATH,
        texts,
        device,
    )

    # ========================================================
    # SEARCH
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
        v2_corpus,
    )

    print(
        "V3 Search..."
    )

    (
        v3_indices,
        v3_scores,
    ) = retrieve(
        v3_queries,
        v3_corpus,
    )

    # ========================================================
    # LABEL METRICS
    # ========================================================

    v2_metrics = label_metrics(
        dev_df,
        corpus_df,
        v2_indices,
    )

    v3_metrics = label_metrics(
        dev_df,
        corpus_df,
        v3_indices,
    )

    # ========================================================
    # ANSWER JUDGE
    # ========================================================

    print()
    print("=" * 70)
    print("ANSWER RELEVANCE")
    print("=" * 70)

    judge = SentenceTransformer(
        JUDGE_MODEL,
        device=device,
    )

    corpus_answers = (
        corpus_df["answer"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    gold_answers = (
        dev_df["answer"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        "Corpus Answer Embedding..."
    )

    corpus_answer_embeddings = (
        judge.encode(
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
        "Gold Answer Embedding..."
    )

    gold_answer_embeddings = (
        judge.encode(
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
    # NON-TIE ANSWER WIN RATE
    # ========================================================

    delta = (
        v3_answer_best3
        - v2_answer_best3
    )

    epsilon = 1e-4

    v3_wins = int(
        (
            delta
            > epsilon
        ).sum()
    )

    v2_wins = int(
        (
            delta
            < -epsilon
        ).sum()
    )

    ties = int(
        (
            np.abs(delta)
            <= epsilon
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

        v3_non_tie_win_rate = 0.0

    # ========================================================
    # RESULT
    # ========================================================

    result = f"""
V2 VS V3 DEV EVALUATION
======================================================================

Dev Samples:
{len(dev_df)}


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

Answer Best@3 Delta:
{np.mean(v3_answer_best3) - np.mean(v2_answer_best3):+.6f}


ANSWER BEST@3 WIN RATE
----------------------------------------------------------------------

V3 Wins:
{v3_wins}

V2 Wins:
{v2_wins}

Ties:
{ties}

V3 Non-Tie Win Rate:
{v3_non_tie_win_rate:.4f}
"""

    print(
        result
    )

    REPORT_PATH.write_text(
        result,
        encoding="utf-8",
    )

    detail_df = pd.DataFrame(
        {
            "subject": (
                dev_df["subject"]
            ),

            "type": (
                dev_df["type"]
            ),

            "queue": (
                dev_df["queue"]
            ),

            "priority": (
                dev_df["priority"]
            ),

            "v2_top1_similarity": (
                v2_scores[:, 0]
            ),

            "v3_top1_similarity": (
                v3_scores[:, 0]
            ),

            "v2_answer_best3": (
                v2_answer_best3
            ),

            "v3_answer_best3": (
                v3_answer_best3
            ),

            "answer_delta": (
                delta
            ),
        }
    )

    detail_df.to_csv(
        DETAIL_PATH,
        index=False,
    )

    print()
    print(
        f"Report:\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()