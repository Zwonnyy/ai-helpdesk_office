from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_train.csv"
)

PAIR_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "semantic_pairs.csv"
)

MODEL_FILES = {

    "Pretrained": (
        BASE_DIR
        / "models"
        / "rag_embeddings.npy"
    ),

    "V1": (
        BASE_DIR
        / "models"
        / "rag_embeddings_finetuned.npy"
    ),

    "V2": (
        BASE_DIR
        / "models"
        / "rag_embeddings_finetuned_v2.npy"
    ),

    "V3": (
        BASE_DIR
        / "models"
        / "rag_embeddings_finetuned_v3.npy"
    ),
}

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "embedding_space_diagnostics.txt"
)


RANDOM_SAMPLE_SIZE = 4000
PAIR_SAMPLE_SIZE = 5000

SEED = 42


# ============================================================
# NORMALIZE
# ============================================================

def normalize(
    embeddings,
):

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.clip(
        norms,
        1e-12,
        None,
    )

    return (
        embeddings
        / norms
    )


# ============================================================
# EFFECTIVE RANK
# ============================================================

def effective_rank(
    embeddings,
):

    # 중심화
    centered = (
        embeddings
        - embeddings.mean(
            axis=0,
            keepdims=True,
        )
    )

    # singular values
    singular_values = (
        np.linalg.svd(
            centered,
            full_matrices=False,
            compute_uv=False,
        )
    )

    eigenvalues = (
        singular_values ** 2
    )

    total = (
        eigenvalues.sum()
    )

    if total <= 0:
        return 0.0

    probabilities = (
        eigenvalues
        / total
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities
        )
    )

    return float(
        np.exp(
            entropy
        )
    )


# ============================================================
# PARTICIPATION RATIO
# ============================================================

def participation_ratio(
    embeddings,
):

    centered = (
        embeddings
        - embeddings.mean(
            axis=0,
            keepdims=True,
        )
    )

    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )

    eigenvalues = (
        singular_values ** 2
    )

    numerator = (
        eigenvalues.sum() ** 2
    )

    denominator = (
        np.square(
            eigenvalues
        ).sum()
    )

    if denominator <= 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EMBEDDING SPACE DIAGNOSTICS")
    print("=" * 70)

    rng = np.random.default_rng(
        SEED
    )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    pair_df = pd.read_csv(
        PAIR_PATH
    )

    # --------------------------------------------------------
    # text -> corpus index
    # --------------------------------------------------------

    text_to_index = {}

    for index, text in enumerate(
        train_df[
            "text"
        ]
        .fillna("")
        .astype(str)
    ):

        if text not in text_to_index:

            text_to_index[
                text
            ] = index

    # --------------------------------------------------------
    # Positive pairs
    # --------------------------------------------------------

    positive_pairs = []

    for _, row in pair_df.iterrows():

        anchor = str(
            row["anchor"]
        )

        positive = str(
            row["positive"]
        )

        if (
            anchor in text_to_index
            and positive in text_to_index
        ):

            positive_pairs.append(
                (
                    text_to_index[
                        anchor
                    ],
                    text_to_index[
                        positive
                    ],
                )
            )

    if (
        len(positive_pairs)
        > PAIR_SAMPLE_SIZE
    ):

        selected = rng.choice(
            len(positive_pairs),
            size=PAIR_SAMPLE_SIZE,
            replace=False,
        )

        positive_pairs = [
            positive_pairs[i]
            for i in selected
        ]

    positive_a = np.array(
        [
            pair[0]
            for pair in positive_pairs
        ]
    )

    positive_b = np.array(
        [
            pair[1]
            for pair in positive_pairs
        ]
    )

    # --------------------------------------------------------
    # Clean negative pairs
    #
    # Queue와 Type이 모두 다른 샘플
    # --------------------------------------------------------

    negative_pairs = []

    total = len(
        train_df
    )

    while (
        len(negative_pairs)
        < PAIR_SAMPLE_SIZE
    ):

        a = int(
            rng.integers(
                0,
                total
            )
        )

        b = int(
            rng.integers(
                0,
                total
            )
        )

        if a == b:
            continue

        row_a = train_df.iloc[
            a
        ]

        row_b = train_df.iloc[
            b
        ]

        if (
            row_a["queue"]
            == row_b["queue"]
        ):
            continue

        if (
            row_a["type"]
            == row_b["type"]
        ):
            continue

        negative_pairs.append(
            (
                a,
                b,
            )
        )

    negative_a = np.array(
        [
            pair[0]
            for pair in negative_pairs
        ]
    )

    negative_b = np.array(
        [
            pair[1]
            for pair in negative_pairs
        ]
    )

    # --------------------------------------------------------
    # Random sample for geometry
    # --------------------------------------------------------

    sample_size = min(
        RANDOM_SAMPLE_SIZE,
        total,
    )

    sample_indices = rng.choice(
        total,
        size=sample_size,
        replace=False,
    )

    results = []

    for name, path in MODEL_FILES.items():

        if not path.exists():

            print(
                f"{name}: 파일 없음 - SKIP"
            )

            continue

        print()
        print(
            f"[{name}] 분석..."
        )

        embeddings = np.load(
            path
        ).astype(
            np.float32
        )

        embeddings = normalize(
            embeddings
        )

        sampled = embeddings[
            sample_indices
        ]

        # ----------------------------------------------------
        # Random pair cosine
        # ----------------------------------------------------

        random_a = rng.integers(
            0,
            sample_size,
            size=PAIR_SAMPLE_SIZE,
        )

        random_b = rng.integers(
            0,
            sample_size,
            size=PAIR_SAMPLE_SIZE,
        )

        random_similarity = np.sum(
            sampled[
                random_a
            ]
            * sampled[
                random_b
            ],
            axis=1,
        )

        # ----------------------------------------------------
        # Positive
        # ----------------------------------------------------

        positive_similarity = np.sum(
            embeddings[
                positive_a
            ]
            * embeddings[
                positive_b
            ],
            axis=1,
        )

        # ----------------------------------------------------
        # Negative
        # ----------------------------------------------------

        negative_similarity = np.sum(
            embeddings[
                negative_a
            ]
            * embeddings[
                negative_b
            ],
            axis=1,
        )

        # ----------------------------------------------------
        # Dimension variance
        # ----------------------------------------------------

        dimension_variance = np.var(
            sampled,
            axis=0,
        )

        mean_dimension_variance = float(
            np.mean(
                dimension_variance
            )
        )

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        eff_rank = effective_rank(
            sampled
        )

        part_ratio = participation_ratio(
            sampled
        )

        result = {

            "model": name,

            "random_mean": float(
                np.mean(
                    random_similarity
                )
            ),

            "random_std": float(
                np.std(
                    random_similarity
                )
            ),

            "positive_mean": float(
                np.mean(
                    positive_similarity
                )
            ),

            "negative_mean": float(
                np.mean(
                    negative_similarity
                )
            ),

            "positive_negative_gap": float(
                np.mean(
                    positive_similarity
                )
                - np.mean(
                    negative_similarity
                )
            ),

            "dimension_variance_mean": (
                mean_dimension_variance
            ),

            "effective_rank": (
                eff_rank
            ),

            "participation_ratio": (
                part_ratio
            ),
        }

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC RESULT")
    print("=" * 70)

    print(
        result_df.to_string(
            index=False
        )
    )

    report = (
        "EMBEDDING SPACE DIAGNOSTICS\n"
        + "=" * 100
        + "\n\n"
        + result_df.to_string(
            index=False
        )
        + "\n\n"
        + "Interpretation:\n"
        + "- Random cosine가 지나치게 높으면 "
          "전체 표현 공간 압축을 의심할 수 있음.\n"
        + "- Positive-Negative Gap은 클수록 "
          "유사/비유사 구분이 잘 되는 방향.\n"
        + "- Effective Rank / Participation Ratio가 "
          "극단적으로 낮아지면 차원 활용 축소를 의심할 수 있음.\n"
        + "- 단일 지표만으로 collapse를 확정하지 않고 "
          "retrieval 성능과 함께 판단해야 함.\n"
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(
        f"Report:\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()