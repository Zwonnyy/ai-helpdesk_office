from pathlib import Path

import pandas as pd
import torch

from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from sentence_transformers.training_args import (
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
    / "rag_train.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_training"
)

FINAL_MODEL_DIR = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_model"
)


# ============================================================
# MODEL
# ============================================================

BASE_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ============================================================
# DATA
# ============================================================

def prepare_dataset():

    print("=" * 70)
    print("HELPDESK EMBEDDING DATA PREPARATION")
    print("=" * 70)

    df = pd.read_csv(
        TRAIN_PATH
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

    df = df[
        (df["text"].str.len() > 0)
        & (df["queue"].str.len() > 0)
        & (df["type"].str.len() > 0)
    ].copy()

    # --------------------------------------------------------
    # queue + type을 하나의 학습 label로 사용
    # --------------------------------------------------------

    df["group_label"] = (
        df["queue"]
        + " || "
        + df["type"]
    )

    # Triplet 학습에는 같은 label 샘플이
    # 최소 2개 이상 필요
    counts = (
        df["group_label"]
        .value_counts()
    )

    valid_labels = counts[
        counts >= 2
    ].index

    df = df[
        df["group_label"].isin(
            valid_labels
        )
    ].copy()

    # --------------------------------------------------------
    # 문자열 label -> 정수
    # --------------------------------------------------------

    encoder = LabelEncoder()

    df["label"] = (
        encoder.fit_transform(
            df["group_label"]
        )
    )

    print(
        f"Training samples : {len(df):,}"
    )

    print(
        f"Label groups     : {df['label'].nunique():,}"
    )

    print()
    print(
        "Group distribution:"
    )

    print(
        df["group_label"]
        .value_counts()
        .head(20)
    )

    # --------------------------------------------------------
    # Sentence Transformers Dataset
    # --------------------------------------------------------

    dataset = Dataset.from_dict(
        {
            "sentence": (
                df["text"].tolist()
            ),

            "label": (
                df["label"]
                .astype(int)
                .tolist()
            ),
        }
    )

    return dataset


# ============================================================
# TRAIN
# ============================================================

def main():

    dataset = prepare_dataset()

    print()
    print("=" * 70)
    print("HELPDESK EMBEDDING FINE-TUNING")
    print("=" * 70)

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
                torch.cuda.get_device_properties(
                    0
                ).total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    # --------------------------------------------------------
    # Base model
    # --------------------------------------------------------

    print()
    print(
        "Base Model 로드..."
    )

    model = SentenceTransformer(
        BASE_MODEL,
        device=device,
    )

    # 너무 긴 Ticket 때문에
    # VRAM이 과도하게 사용되는 것 방지
    model.max_seq_length = 256

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss = (
        losses.BatchHardSoftMarginTripletLoss(
            model=model
        )
    )

    # --------------------------------------------------------
    # Training Arguments
    # --------------------------------------------------------

    args = (
        SentenceTransformerTrainingArguments(

            output_dir=str(
                OUTPUT_DIR
            ),

            num_train_epochs=1,

            per_device_train_batch_size=16,

            learning_rate=2e-5,

            warmup_ratio=0.1,

            weight_decay=0.01,

            fp16=(
                torch.cuda.is_available()
            ),

            bf16=False,

            # Triplet Loss 핵심
            batch_sampler=(
                BatchSamplers.GROUP_BY_LABEL
            ),

            logging_steps=100,

            save_strategy="epoch",

            save_total_limit=1,

            report_to="none",

            seed=42,

            data_seed=42,
        )
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

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
    print("TRAINING START")
    print("=" * 70)

    trainer.train()

    # --------------------------------------------------------
    # Final Model Save
    # --------------------------------------------------------

    model.save_pretrained(
        str(
            FINAL_MODEL_DIR
        )
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Fine-tuned model:"
    )

    print(
        FINAL_MODEL_DIR
    )


if __name__ == "__main__":
    main()