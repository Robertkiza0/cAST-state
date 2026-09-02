"""Two-stage retriever: CodeSage dense retrieval for a candidate pool
(--candidate-pool, default 20), then a reranking pass using a scope-match
heuristic — self.* attribute overlap between the query's currently-active
state and each candidate's cast_scope header, plus a small decorator bonus.

Honesty check before reading too much into this: this reranking score is
conceptually close to a weighted-attention scoring signal already tested
and found NOT to significantly beat raw-text matching on a different axis
in this project's earlier work (repocoder-mine's weighted_ast_scorer,
confirmed via McNemar on both RepoCoder and CCEval — see
colab-experiment-findings memory). This is a genuinely different setting
though: a RERANKING stage over dense retrieval results, not the primary
retrieval signal itself, which is worth checking on its own rather than
assuming the earlier null result carries over — but go in expecting that
possibility, not assuming a win.
"""

import re

from .codesage_retriever import CodeSageRetriever

_SELF_ATTR_RE = re.compile(r"self\.(\w+)")


def extract_query_scope_vars(unfinished_code: str, tail_lines: int = 15) -> list[str]:
    """`self.<name>` mentions in the last `tail_lines` lines of the code
    written so far — a cheap proxy for "state plausibly relevant near the
    cursor", not a real scope analysis (unlike astchunk_scope's own
    self.* extraction, which walks the actual enclosing class at CHUNKING
    time — this one only looks at literal token mentions in the QUERY
    text, at retrieval time, since we have no cursor-scope tracker in this
    project yet)."""
    tail = "\n".join(unfinished_code.splitlines()[-tail_lines:])
    return sorted(set(_SELF_ATTR_RE.findall(tail)))


class AgenticScopeReranker:
    def __init__(
        self, chunks: list[dict], candidate_pool: int = 20, model_name: str = "codesage/codesage-small-v2",
        dense_retriever: object | None = None,
    ):
        # dense_retriever est injectable (pour les tests, avec un double qui
        # n'a pas besoin de torch) ; sinon, CodeSage réel est construit ici.
        self.dense = dense_retriever if dense_retriever is not None else CodeSageRetriever(chunks, model_name=model_name)
        self.candidate_pool = candidate_pool

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        candidates = self.dense.retrieve(query, k=self.candidate_pool)
        query_vars = extract_query_scope_vars(query)

        scored = []
        for chunk in candidates:
            score = 1.0
            header = chunk.get("header", "")
            for var in query_vars:
                if f"self.{var}" in header:
                    score += 1.5  # correspondance exacte d'attribut d'état
            if "@" in header:
                score += 0.5  # décorateur présent dans l'ancêtre
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]
