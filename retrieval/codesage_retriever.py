"""CodeSage dense retriever (codesage/codesage-small-v2) — one of the THREE
retrievers actually evaluated in the cAST paper itself (BGE-base, GIST-base,
CodeSage-small-v2; see cast_scope_paper_methodology_notes memory on why
retrieval/bm25_retriever.py's BM25 departs from all three). Adding this one
specifically closes that gap for at least one retriever choice.

Needs torch + transformers; NOT testable on this development machine (no
torch installed here — same limitation as generation/generator.py's
HFGenerator). `trust_remote_code=True` is required by this model's custom
modeling code on the Hub — a real trust decision (arbitrary code execution
from the model repo), not hidden here, just necessary for this specific
model to load at all.
"""

import numpy as np


def _patch_conv1d_compat() -> None:
    """CodeSage's custom modeling code (loaded via trust_remote_code=True,
    pinned to whatever revision the Hub serves) does
    `from transformers.modeling_utils import Conv1D` — valid in the
    transformers version that code was written against, but recent
    transformers releases moved Conv1D to transformers.pytorch_utils and
    modeling_utils no longer re-exports it at all, so that import fails
    outright (ImportError, not something try/except around model loading
    can route around, since it happens inside the remote module's own
    top-level import statements). Re-exposing the class under its old
    location before loading the model is the standard workaround for a
    remote-code repo written against an older transformers API — this
    doesn't change any behavior, Conv1D itself is unchanged, only where
    it's importable from."""
    import transformers.modeling_utils as modeling_utils

    if not hasattr(modeling_utils, "Conv1D"):
        from transformers.pytorch_utils import Conv1D
        modeling_utils.Conv1D = Conv1D


class CodeSageRetriever:
    # Cached at class level (tokenizer + model), keyed by model_name: this
    # class gets instantiated fresh per (repo, task) in run_benchmark.py's
    # loop — reloading a real embedding model from disk/HF cache every time
    # would be needlessly slow and, on Colab, adds its own crash surface.
    _cache: dict[str, tuple] = {}

    def __init__(self, chunks: list[dict], model_name: str = "codesage/codesage-small-v2"):
        import torch  # noqa: F401 - présence vérifiée tôt, message d'erreur clair sinon
        from transformers import AutoModel, AutoTokenizer

        if model_name not in CodeSageRetriever._cache:
            _patch_conv1d_compat()
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            model.eval()
            CodeSageRetriever._cache[model_name] = (tokenizer, model)
        self.tokenizer, self.model = CodeSageRetriever._cache[model_name]

        self.chunks = chunks
        # header+content, même principe que BM25Retriever : le texte enrichi
        # de cast_scope doit rester disponible au retriever, même si la
        # génération ne le voit que dans une section séparée (voir
        # chunkers/__init__.py).
        texts = [f"{chunk.get('header', '')}\n{chunk['content']}" for chunk in chunks]
        self.embeddings = self._encode(texts) if texts else np.zeros((0, 1))

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling sur les tokens (choix simple, cohérent avec l'usage
        # courant de ce modèle en l'absence d'un pooling dédié exposé par
        # l'API du modèle).
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if len(self.chunks) == 0:
            return []
        query_vec = self._encode([query])[0]
        chunk_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = chunk_norms * query_norm
        denom[denom == 0] = 1e-9
        scores = (self.embeddings @ query_vec) / denom
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]
