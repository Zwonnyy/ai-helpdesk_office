from pathlib import Path

import pandas as pd
import torch

from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
)

from sentence_transformers.sentence_transformer.training_args import (
    BatchSamplers,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "v3_hard_negative_triplets.csv"
)

BASE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_model_v2"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_v3_training"
)

FINAL_MODEL_DIR = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_model_v3"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 70)
    print("V3 TRAINING DATA LOAD")
    print("=" * 70)

    df = pd.read_csv(
        TRAIN_PATH
    )

    expected = [
        "anchor",
        "positive",
        "negative_1",
        "negative_2",
        "negative_3",
    ]

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"필수 컬럼이 없습니다: {missing}\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    df = df[
        expected
    ].copy()

    for column in expected:

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    valid_mask = True

    for column in expected:

        valid_mask = (
            valid_mask
            & (
                df[column]
                .str.len()
                > 0
            )
        )

    df = (
        df[
            valid_mask
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    print(
        f"Training Samples: {len(df):,}"
    )

    dataset = Dataset.from_dict(
        {
            "anchor": (
                df["anchor"]
                .tolist()
            ),

            "positive": (
                df["positive"]
                .tolist()
            ),

            "negative_1": (
                df["negative_1"]
                .tolist()
            ),

            "negative_2": (
                df["negative_2"]
                .tolist()
            ),

            "negative_3": (
                df["negative_3"]
                .tolist()
            ),
        }
    )

    return dataset


# ============================================================
# MAIN
# ============================================================

def main():

    dataset = load_dataset()

    print()
    print("=" * 70)
    print("HELPDESK EMBEDDING V3")
    print("=" * 70)

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

        print(
            "VRAM:",
            round(
                torch.cuda
                .get_device_properties(
                    0
                )
                .total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    # --------------------------------------------------------
    # START FROM V2
    # --------------------------------------------------------

    print()
    print(
        "V2 Model 로드..."
    )

    model = SentenceTransformer(
        str(
            BASE_MODEL_PATH
        ),
        device=device,
    )

    model.max_seq_length = 192

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss = (
        CachedMultipleNegativesRankingLoss(

            model=model,

            mini_batch_size=4,
        )
    )

    # --------------------------------------------------------
    # Training
    #
    # V2를 망가뜨리지 않도록
    # 매우 보수적인 LR 사용
    # --------------------------------------------------------

    args = (
        SentenceTransformerTrainingArguments(

            output_dir=str(
                OUTPUT_DIR
            ),

            num_train_epochs=1,

            per_device_train_batch_size=32,

            learning_rate=5e-6,

            weight_decay=0.01,

            # transformers v5에서는
            # 0~1 float를 ratio로 사용 가능
            warmup_steps=0.05,

            fp16=(
                torch.cuda.is_available()
            ),

            bf16=False,

            batch_sampler=(
                BatchSamplers
                .NO_DUPLICATES
            ),

            logging_steps=25,

            save_strategy="epoch",

            save_total_limit=1,

            report_to="none",

            seed=42,

            data_seed=42,
        )
    )

    trainer = (
        SentenceTransformerTrainer(

            model=model,

            args=args,

            train_dataset=dataset,

            loss=loss,
        )
    )

    print()
    print("=" * 70)
    print("V3 TRAINING START")
    print("=" * 70)

    trainer.train()

    model.save_pretrained(
        str(
            FINAL_MODEL_DIR
        )
    )

    print()
    print("=" * 70)
    print("V3 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Model:\n{FINAL_MODEL_DIR}"
    )


if __name__ == "__main__":
    main()