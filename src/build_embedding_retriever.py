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

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EMBEDDING_PATH = (
    MODEL_DIR
    / "rag_embeddings.npy"
)

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def main():

    print("=" * 70)
    print("SEMANTIC RETRIEVER BUILD")
    print("=" * 70)

    # --------------------------------------------------------
    # 데이터
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    texts = (
        train_df["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print(
        f"Knowledge Base: {len(texts):,}건"
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
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Embedding Model 로드..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    print(
        "Embedding 생성 시작..."
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embeddings = model.encode(
        texts,

        batch_size=64,

        show_progress_bar=True,

        normalize_embeddings=True,

        convert_to_numpy=True,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    np.save(
        EMBEDDING_PATH,
        embeddings,
    )

    print()
    print("=" * 70)
    print("SEMANTIC RETRIEVER READY")
    print("=" * 70)

    print(
        f"Shape: {embeddings.shape}"
    )

    print(
        f"저장 위치:"
        f"\n{EMBEDDING_PATH}"
    )


if __name__ == "__main__":
    main()