from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schema import RetrievedPolicy


class PolicyRAG:
    def __init__(self, docs_dir: str | Path):
        self.docs_dir = Path(docs_dir)
        self.documents: list[dict[str, str]] = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.matrix = None
        self._load_documents()
        self._build_index()

    def _load_documents(self) -> None:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Policy docs directory not found: {self.docs_dir}")

        paths = sorted(self.docs_dir.glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"No markdown policy documents found in: {self.docs_dir}")

        for path in paths:
            text = path.read_text(encoding="utf-8")
            clean_text = re.sub(r"\s+", " ", text).strip()
            self.documents.append(
                {
                    "source_id": path.name,
                    "text": clean_text,
                }
            )

    def _build_index(self) -> None:
        texts = [doc["text"] for doc in self.documents]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        if self.matrix is None:
            raise RuntimeError("RAG index was not built")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            doc = self.documents[int(idx)]
            results.append(
                RetrievedPolicy(
                    source_id=doc["source_id"],
                    score=float(scores[int(idx)]),
                    text=doc["text"][:1200],
                )
            )

        return results

    def source_ids(self) -> set[str]:
        return {doc["source_id"] for doc in self.documents}
