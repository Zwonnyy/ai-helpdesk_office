from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[1]

CORPUS_PATH = (
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

EMBEDDING_PATH = (
    BASE_DIR
    / "models"
    / "rag_embeddings_finetuned_v3.npy"
)


class AnswerRetriever:

    def __init__(self):

        self.model = None

        self.embeddings = None

        self.corpus = None

        self.loaded = False

        # 일단 GPU 사용
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load(self):

        print()
        print("=" * 70)
        print("SEMANTIC ANSWER RETRIEVER LOADING")
        print("=" * 70)

        if not CORPUS_PATH.exists():

            raise FileNotFoundError(
                f"RAG corpus가 없습니다.\n"
                f"{CORPUS_PATH}"
            )

        if not EMBEDDING_PATH.exists():

            raise FileNotFoundError(
                f"Embedding 파일이 없습니다.\n"
                f"{EMBEDDING_PATH}\n\n"
                "먼저 실행하세요:\n"
                "uv run python "
                "src/build_embedding_retriever.py"
            )

        # ----------------------------------------------------
        # Corpus
        # ----------------------------------------------------

        self.corpus = pd.read_csv(
            CORPUS_PATH
        )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        self.embeddings = np.load(
            EMBEDDING_PATH
        )

        if (
            len(self.corpus)
            != len(self.embeddings)
        ):
            raise ValueError(
                "Corpus와 Embedding 개수가 다릅니다.\n"
                f"Corpus: {len(self.corpus)}\n"
                f"Embeddings: {len(self.embeddings)}"
            )

        # ----------------------------------------------------
        # Sentence Transformer
        # ----------------------------------------------------

        print(
            f"Device: {self.device}"
        )

        if torch.cuda.is_available():
            print(
                "GPU:",
                torch.cuda.get_device_name(0),
            )

        print(
            "Sentence Transformer 로드..."
        )

        self.model = SentenceTransformer(
            str(MODEL_PATH),
            device=self.device,
        )

        self.loaded = True

        print(
            f"Corpus: {len(self.corpus):,}건"
        )

        print(
            f"Embedding Matrix: "
            f"{self.embeddings.shape}"
        )

        print(
            "SEMANTIC RETRIEVER READY"
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        subject: str,
        body: str,
        top_k: int = 3,
    ):

        if not self.loaded:

            raise RuntimeError(
                "AnswerRetriever가 "
                "로드되지 않았습니다."
            )

        # ----------------------------------------------------
        # Query text
        # ----------------------------------------------------

        text = (
            subject.strip()
            + "\n"
            + body.strip()
        )

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_embedding = (
            self.model.encode(
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
            .astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # Cosine similarity
        #
        # normalize_embeddings=True 이므로
        # dot product == cosine similarity
        # ----------------------------------------------------

        similarities = (
            self.embeddings
            @ query_embedding
        )

        top_k = min(
            top_k,
            len(similarities),
        )

        # ----------------------------------------------------
        # Top K
        # ----------------------------------------------------

        candidate_indices = (
            np.argpartition(
                similarities,
                -top_k,
            )[-top_k:]
        )

        top_indices = (
            candidate_indices[
                np.argsort(
                    similarities[
                        candidate_indices
                    ]
                )[::-1]
            ]
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        results = []

        for index in top_indices:

            row = self.corpus.iloc[
                index
            ]

            results.append(
                {
                    "score": float(
                        similarities[index]
                    ),

                    "subject": str(
                        row["subject"]
                    ),

                    "body": str(
                        row["body"]
                    ),

                    "answer": str(
                        row["answer"]
                    ),

                    "type": str(
                        row["type"]
                    ),

                    "queue": str(
                        row["queue"]
                    ),

                    "priority": str(
                        row["priority"]
                    ),

                    "language": str(
                        row["language"]
                    ),
                }
            )

        return results