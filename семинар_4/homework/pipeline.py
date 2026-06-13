from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer


Strategy = Literal["fixed", "recursive"]


@dataclass
class Document:
    source: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


class RAGPipeline:
    def __init__(
        self,
        data_dir: str | Path = "data",
        strategy: Strategy = "fixed",
        top_k: int = 5,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.strategy = strategy
        self.top_k = top_k
        self.model = SentenceTransformer(model_name)

        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.chunk_embeddings: np.ndarray | None = None

    def load_documents(self) -> list[Document]:
        paths = sorted(self.data_dir.glob("doc_*.md"))
        if not paths:
            raise FileNotFoundError(f"No doc_*.md files found in {self.data_dir}")

        documents: list[Document] = []

        for path in paths:
            source = path.stem
            text = path.read_text(encoding="utf-8")
            documents.append(Document(source=source, text=text))

        self.documents = documents
        return documents

    @staticmethod
    def chunk_fixed(text: str, chunk_size: int = 2000) -> list[str]:
        return [
            text[i : i + chunk_size]
            for i in range(0, len(text), chunk_size)
            if text[i : i + chunk_size].strip()
        ]

    @staticmethod
    def chunk_recursive_like(
        text: str,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ) -> list[str]:
        """
        Умная стратегия без дополнительной зависимости langchain:
        сначала старается сохранять абзацы, потом режет слишком длинные
        фрагменты с перекрытием.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                if current.strip():
                    chunks.append(current.strip())
                    current = ""

                start = 0
                step = max(1, chunk_size - chunk_overlap)
                while start < len(paragraph):
                    part = paragraph[start : start + chunk_size]
                    if part.strip():
                        chunks.append(part.strip())
                    start += step
                continue

            candidate = (current + "\n\n" + paragraph).strip() if current else paragraph

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = paragraph

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def build_chunks(self) -> list[Chunk]:
        if not self.documents:
            self.load_documents()

        chunks: list[Chunk] = []

        for document in self.documents:
            if self.strategy == "fixed":
                texts = self.chunk_fixed(document.text, chunk_size=2000)
            elif self.strategy == "recursive":
                texts = self.chunk_recursive_like(
                    document.text,
                    chunk_size=400,
                    chunk_overlap=80,
                )
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            for idx, chunk_text in enumerate(texts):
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.source}_chunk_{idx}",
                        source=document.source,
                        text=chunk_text,
                    )
                )

        self.chunks = chunks
        return chunks

    def build_index(self) -> None:
        if not self.chunks:
            self.build_chunks()

        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.chunk_embeddings = embeddings

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        if self.chunk_embeddings is None:
            self.build_index()

        if top_k is None:
            top_k = self.top_k

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.chunk_embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict] = []

        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[int(idx)]),
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "text": chunk.text,
                }
            )

        return results
