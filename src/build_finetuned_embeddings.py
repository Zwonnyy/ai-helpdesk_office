from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


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

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "helpdesk_embedding_model"
)

OUTPUT_PATH = (
    BASE_DIR
    / "models"
    / "rag_embeddings_finetuned.npy"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINE-TUNED EMBEDDING BUILD")
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
    # 모델 확인
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Fine-tuned 모델이 없습니다.\n"
            f"{MODEL_PATH}\n\n"
            "먼저 실행하세요:\n"
            "uv run python "
            "src/train_helpdesk_embedding.py"
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
    # Fine-tuned Model
    # --------------------------------------------------------

    print()
    print(
        "Fine-tuned Model 로드..."
    )

    model = SentenceTransformer(
        str(MODEL_PATH),
        device=device,
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    print(
        "Fine-tuned Embedding 생성 시작..."
    )

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
        OUTPUT_PATH,
        embeddings,
    )

    print()
    print("=" * 70)
    print("FINE-TUNED EMBEDDING READY")
    print("=" * 70)

    print(
        f"Shape: {embeddings.shape}"
    )

    print(
        "저장 위치:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
