from pathlib import Path

import pandas as pd
import torch

from datasets import Dataset

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

PAIR_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "semantic_pairs.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_v2_training"
)

FINAL_MODEL_DIR = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_model_v2"
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

def load_dataset():

    print("=" * 70)
    print("SEMANTIC PAIR DATA LOAD")
    print("=" * 70)

    df = pd.read_csv(
        PAIR_PATH
    )

    df["anchor"] = (
        df["anchor"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["positive"] = (
        df["positive"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["anchor"].str.len() > 0)
        & (df["positive"].str.len() > 0)
    ].copy()

    df = df[
        df["anchor"]
        != df["positive"]
    ].copy()

    df = df.drop_duplicates(
        subset=[
            "anchor",
            "positive",
        ]
    )

    df = df.reset_index(
        drop=True
    )

    print(
        f"Training Pairs: {len(df):,}"
    )

    print(
        f"Mean Similarity: "
        f"{df['similarity'].mean():.4f}"
    )

    # --------------------------------------------------------
    # HuggingFace Dataset
    #
    # 중요:
    # MNRL에는 label이 필요 없다.
    # anchor / positive만 전달한다.
    # --------------------------------------------------------

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
        }
    )

    return dataset


# ============================================================
# TRAIN
# ============================================================

def main():

    dataset = load_dataset()

    print()
    print("=" * 70)
    print("HELPDESK EMBEDDING V2 TRAINING")
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
                torch.cuda.get_device_properties(
                    0
                ).total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Base Model 로드..."
    )

    model = SentenceTransformer(
        BASE_MODEL,
        device=device,
    )

    # VRAM 절약
    model.max_seq_length = 192

    # --------------------------------------------------------
    # LOSS
    #
    # 큰 batch의 효과를 얻되
    # forward는 작은 mini-batch로 수행
    # --------------------------------------------------------

    loss = (
        losses.CachedMultipleNegativesRankingLoss(

            model=model,

            mini_batch_size=8,
        )
    )

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    args = (
        SentenceTransformerTrainingArguments(

            output_dir=str(
                OUTPUT_DIR
            ),

            num_train_epochs=1,

            # Cached MNRL이므로
            # 논리적 Batch는 크게
            per_device_train_batch_size=64,

            learning_rate=1e-5,

            warmup_ratio=0.1,

            weight_decay=0.01,

            fp16=(
                torch.cuda.is_available()
            ),

            bf16=False,

            # MNRL의 in-batch negative에
            # 같은 문장이 들어가지 않도록 함
            batch_sampler=(
                BatchSamplers.NO_DUPLICATES
            ),

            logging_steps=25,

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

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING START")
    print("=" * 70)

    trainer.train()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    model.save_pretrained(
        str(
            FINAL_MODEL_DIR
        )
    )

    print()
    print("=" * 70)
    print("TRAINING V2 COMPLETE")
    print("=" * 70)

    print(
        "Model saved:"
    )

    print(
        FINAL_MODEL_DIR
    )


if __name__ == "__main__":
    main()