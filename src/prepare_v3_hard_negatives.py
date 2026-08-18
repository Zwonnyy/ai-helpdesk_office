from pathlib import Path

import pandas as pd
import torch

from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
)

from sentence_transformers.util import (
    mine_hard_negatives,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PAIR_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "semantic_pairs.csv"
)

TRAIN_PATH = (
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

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "v3_hard_negative_triplets.csv"
)

CACHE_DIR = (
    BASE_DIR
    / "models"
    / "hard_negative_cache"
)


# ============================================================
# CONFIG
# ============================================================

NUM_NEGATIVES = 3

RANGE_MIN = 1
RANGE_MAX = 50

MIN_SCORE = 0.55
MAX_SCORE = 0.95

RELATIVE_MARGIN = 0.05

BATCH_SIZE = 64


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V3 HARD NEGATIVE MINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Positive Pair
    # --------------------------------------------------------

    pair_df = pd.read_csv(
        PAIR_PATH
    )

    pair_df = pair_df[
        [
            "anchor",
            "positive",
        ]
    ].copy()

    pair_df["anchor"] = (
        pair_df["anchor"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    pair_df["positive"] = (
        pair_df["positive"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    pair_df = pair_df[
        (pair_df["anchor"].str.len() > 0)
        & (
            pair_df["positive"]
            .str.len()
            > 0
        )
    ]

    pair_df = (
        pair_df
        .drop_duplicates(
            subset=[
                "anchor",
                "positive",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Positive Pairs: {len(pair_df):,}"
    )

    # --------------------------------------------------------
    # Negative Candidate Corpus
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    corpus = (
        train_df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    corpus = (
        corpus[
            corpus.str.len() > 0
        ]
        .drop_duplicates()
        .tolist()
    )

    print(
        f"Negative Corpus: {len(corpus):,}"
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
    # V2 Model
    #
    # 현재 가장 좋은 Retriever로
    # 자신이 헷갈리는 문서를 찾는다.
    # --------------------------------------------------------

    print()
    print(
        "V2 Model 로드..."
    )

    model = SentenceTransformer(
        str(MODEL_PATH),
        device=device,
    )

    # --------------------------------------------------------
    # HuggingFace Dataset
    # --------------------------------------------------------

    dataset = Dataset.from_dict(
        {
            "anchor": (
                pair_df[
                    "anchor"
                ].tolist()
            ),

            "positive": (
                pair_df[
                    "positive"
                ].tolist()
            ),
        }
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Hard Negative Mining
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MINING START")
    print("=" * 70)

    mined_dataset = (
        mine_hard_negatives(

            dataset=dataset,

            model=model,

            anchor_column_name=(
                "anchor"
            ),

            positive_column_name=(
                "positive"
            ),

            corpus=corpus,

            # Anchor 자기 자신 같은
            # 최상위 후보를 피하기 위함
            range_min=RANGE_MIN,

            range_max=RANGE_MAX,

            # 너무 쉬운 negative 제거
            min_score=MIN_SCORE,

            # Positive일 가능성이 지나치게
            # 높은 문장 제거
            max_score=MAX_SCORE,

            # Positive보다 최소 5% 낮은
            # similarity의 negative만 허용
            relative_margin=(
                RELATIVE_MARGIN
            ),

            num_negatives=(
                NUM_NEGATIVES
            ),

            sampling_strategy="top",

            # 한 row에
            # anchor
            # positive
            # negative_1~3
            output_format="n-tuple",

            batch_size=BATCH_SIZE,

            # 별도 FAISS 설치 없이 진행
            use_faiss=False,

            cache_folder=str(
                CACHE_DIR
            ),

            verbose=True,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    mined_df = (
        mined_dataset
        .to_pandas()
    )

    mined_df = mined_df.dropna()

    mined_df = mined_df.reset_index(
        drop=True
    )

    mined_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("V3 DATASET READY")
    print("=" * 70)

    print(
        f"Samples: {len(mined_df):,}"
    )

    print(
        "Columns:"
    )

    print(
        list(
            mined_df.columns
        )
    )

    print()
    print(
        f"저장 위치:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()