"""Retriever shared by all 3 chunking baselines in run_benchmark.py.

The whole point of the experiment is to isolate the effect of CHUNKING: same
retriever, same generator, only the chunker differs between conditions. BM25
is a reasonable, dependency-light, deterministic choice (no embedding model
to download, no GPU) — ranking is purely a function of each chunk's own text,
so it also naturally rewards cast_scope's chunk_expansion=True headers
(class state / decorators) if and only if that text actually helps match
the query, without any retriever-side special-casing for one condition over
another.
"""

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._corpus_tokens = [tokenize(chunk["content"]) for chunk in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Top-k chunks by BM25 score against query, descending, ties broken
        by original order. Chunks with a non-positive score are dropped
        (BM25Okapi can return 0 or negative for no term overlap at all)."""
        if self._bm25 is None or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(scores, range(len(self.chunks)), self.chunks),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        return [chunk for score, _, chunk in ranked[:k] if score > 0]
