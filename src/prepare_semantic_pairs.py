from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
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

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "semantic_pairs.csv"
)


# ============================================================
# CONFIG
# ============================================================

# 너무 약한 positive pair 제거
MIN_SIMILARITY = 0.80

# 거의 동일 문장 수준의 pair만 학습되는 것도 방지
MAX_SIMILARITY = 0.9995

# 자기 자신 포함해서 몇 개 이웃을 볼지
NEIGHBORS = 8


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEMANTIC POSITIVE PAIR MINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 데이터 로드
    # --------------------------------------------------------

    df = pd.read_csv(
        TRAIN_PATH
    )

    embeddings = np.load(
        EMBEDDING_PATH
    )

    if len(df) != len(embeddings):
        raise ValueError(
            "rag_train과 embedding 개수가 다릅니다.\n"
            f"Train: {len(df)}\n"
            f"Embedding: {len(embeddings)}"
        )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["queue"] = (
        df["queue"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["type"] = (
        df["type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    print(
        f"Train Samples: {len(df):,}"
    )

    print(
        f"Embedding Shape: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # group
    #
    # 같은 Queue + Type 안에서만
    # semantic positive를 찾는다.
    # --------------------------------------------------------

    df["group"] = (
        df["queue"]
        + " || "
        + df["type"]
    )

    groups = df.groupby(
        "group"
    )

    print(
        f"Groups: {df['group'].nunique()}"
    )

    pairs = []

    processed = 0

    # --------------------------------------------------------
    # Group 별 nearest neighbor
    # --------------------------------------------------------

    for group_name, group_df in groups:

        if len(group_df) < 2:
            continue

        global_indices = (
            group_df.index
            .to_numpy()
        )

        group_embeddings = (
            embeddings[
                global_indices
            ]
        )

        n_neighbors = min(
            NEIGHBORS,
            len(group_df),
        )

        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )

        nn.fit(
            group_embeddings
        )

        distances, indices = (
            nn.kneighbors(
                group_embeddings
            )
        )

        # ----------------------------------------------------
        # 각 Anchor별 positive 선택
        # ----------------------------------------------------

        for local_anchor_index in range(
            len(group_df)
        ):

            anchor_global_index = (
                global_indices[
                    local_anchor_index
                ]
            )

            anchor_row = df.loc[
                anchor_global_index
            ]

            anchor_text = (
                anchor_row["text"]
            )

            selected = False

            for neighbor_position in range(
                1,
                n_neighbors,
            ):

                local_positive_index = (
                    indices[
                        local_anchor_index,
                        neighbor_position,
                    ]
                )

                positive_global_index = (
                    global_indices[
                        local_positive_index
                    ]
                )

                positive_row = df.loc[
                    positive_global_index
                ]

                positive_text = (
                    positive_row["text"]
                )

                # 완전히 같은 문장은 제외
                if (
                    anchor_text
                    == positive_text
                ):
                    continue

                similarity = (
                    1.0
                    - distances[
                        local_anchor_index,
                        neighbor_position,
                    ]
                )

                if (
                    similarity
                    < MIN_SIMILARITY
                ):
                    continue

                if (
                    similarity
                    > MAX_SIMILARITY
                ):
                    continue

                pairs.append(
                    {
                        "anchor": anchor_text,
                        "positive": positive_text,

                        "similarity": float(
                            similarity
                        ),

                        "queue": (
                            anchor_row["queue"]
                        ),

                        "type": (
                            anchor_row["type"]
                        ),

                        "group": (
                            group_name
                        ),
                    }
                )

                selected = True
                break

            if not selected:
                continue

        processed += len(
            group_df
        )

        print(
            f"{group_name:<50} "
            f"{len(group_df):>6,} samples"
        )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    pair_df = pd.DataFrame(
        pairs
    )

    if len(pair_df) == 0:
        raise RuntimeError(
            "생성된 Semantic Pair가 없습니다."
        )

    # --------------------------------------------------------
    # 동일 Pair 중복 제거
    # --------------------------------------------------------

    before = len(
        pair_df
    )

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

    removed = (
        before
        - len(pair_df)
    )

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PAIR MINING COMPLETE")
    print("=" * 70)

    print(
        f"Pairs: {len(pair_df):,}"
    )

    print(
        f"Duplicates removed: {removed:,}"
    )

    print(
        f"Mean Similarity: "
        f"{pair_df['similarity'].mean():.4f}"
    )

    print(
        f"Min Similarity : "
        f"{pair_df['similarity'].min():.4f}"
    )

    print(
        f"Max Similarity : "
        f"{pair_df['similarity'].max():.4f}"
    )

    print()
    print(
        "저장 위치:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Sample:"
    )

    print(
        pair_df[
            [
                "queue",
                "type",
                "similarity",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()