from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "rag_train.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_model_v3"
)

OUTPUT_PATH = (
    BASE_DIR
    / "models"
    / "rag_embeddings_finetuned_v3.npy"
)


def main():

    print("=" * 70)
    print("V3 CORPUS EMBEDDING BUILD")
    print("=" * 70)

    df = pd.read_csv(
        TRAIN_PATH
    )

    texts = (
        df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Corpus: {len(texts):,}"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()
    print(
        "V3 Model 로드..."
    )

    model = SentenceTransformer(
        str(MODEL_PATH),
        device=device,
    )

    print(
        "V3 Corpus Embedding 생성..."
    )

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        np.float32
    )

    np.save(
        OUTPUT_PATH,
        embeddings,
    )

    print()
    print("=" * 70)
    print("V3 EMBEDDING READY")
    print("=" * 70)

    print(
        f"Shape: {embeddings.shape}"
    )

    print(
        f"저장 위치:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()