from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import (
    SentenceTransformer,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

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


# ============================================================
# ANSWER RETRIEVER
# ============================================================

class AnswerRetriever:

    def __init__(
        self,
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None

        self.corpus = None

        self.embeddings = None

        self.loaded = False


    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
    ):

        if self.loaded:
            return

        print()
        print("=" * 70)
        print("V3 SEMANTIC ANSWER RETRIEVER")
        print("=" * 70)

        print(
            f"Device: {self.device}"
        )

        # ----------------------------------------------------
        # CORPUS
        # ----------------------------------------------------

        self.corpus = pd.read_csv(
            CORPUS_PATH
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        print(
            "V3 Embedding Model 로드..."
        )

        self.model = (
            SentenceTransformer(
                str(
                    MODEL_PATH
                ),
                device=(
                    self.device
                ),
            )
        )

        # ----------------------------------------------------
        # CORPUS EMBEDDINGS
        # ----------------------------------------------------

        print(
            "V3 Corpus Embeddings 로드..."
        )

        self.embeddings = np.load(
            EMBEDDING_PATH
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if (
            len(self.corpus)
            != len(
                self.embeddings
            )
        ):

            raise ValueError(
                "Corpus와 Embedding 개수가 "
                "일치하지 않습니다.\n"
                f"Corpus={len(self.corpus)} "
                f"Embeddings={len(self.embeddings)}"
            )

        self.loaded = True

        print()
        print(
            f"Corpus: {len(self.corpus):,}"
        )

        print(
            f"Embedding Shape: "
            f"{self.embeddings.shape}"
        )

        print()
        print(
            "V3 SEMANTIC RETRIEVER READY"
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

            self.load()

        # ----------------------------------------------------
        # QUERY
        # ----------------------------------------------------

        query = (
            f"{subject.strip()}\n"
            f"{body.strip()}"
        )

        # ----------------------------------------------------
        # QUERY EMBEDDING
        # ----------------------------------------------------

        query_embedding = (
            self.model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            .astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # COSINE SIMILARITY
        #
        # corpus/query 모두 normalized 상태이므로
        # dot product == cosine similarity
        # ----------------------------------------------------

        similarities = (
            self.embeddings
            @ query_embedding
        )

        top_k = min(
            top_k,
            len(
                similarities
            ),
        )

        # ----------------------------------------------------
        # TOP K
        # ----------------------------------------------------

        candidate_indices = (
            np.argpartition(
                similarities,
                -top_k,
            )[-top_k:]
        )

        sorted_indices = (
            candidate_indices[
                np.argsort(
                    similarities[
                        candidate_indices
                    ]
                )[::-1]
            ]
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        results = []

        for index in sorted_indices:

            row = (
                self.corpus.iloc[
                    index
                ]
            )

            results.append(
                {
                    "score": float(
                        similarities[
                            index
                        ]
                    ),

                    "subject": str(
                        row[
                            "subject"
                        ]
                    ),

                    "body": str(
                        row[
                            "body"
                        ]
                    ),

                    "answer": str(
                        row[
                            "answer"
                        ]
                    ),

                    "type": str(
                        row[
                            "type"
                        ]
                    ),

                    "queue": str(
                        row[
                            "queue"
                        ]
                    ),

                    "priority": str(
                        row[
                            "priority"
                        ]
                    ),

                    "language": str(
                        row[
                            "language"
                        ]
                    ),
                }
            )

        return results