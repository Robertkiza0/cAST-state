"""CodeSage dense retriever (codesage/codesage-small-v2) — one of the THREE
retrievers actually evaluated in the cAST paper itself (BGE-base, GIST-base,
CodeSage-small-v2; see cast_scope_paper_methodology_notes memory on why
retrieval/bm25_retriever.py's BM25 departs from all three). Adding this one
specifically closes that gap for at least one retriever choice.

Loaded via sentence-transformers rather than raw AutoModel/AutoTokenizer:
the model's own card documents both, but raw AutoModel loading hit two
separate transformers-version incompatibilities in a row (Conv1D moved
out of transformers.modeling_utils; then a missing all_tied_weights_keys
attribute deeper in from_pretrained) against this model's aging custom
modeling code (trust_remote_code=True) on a very recent transformers
release — a maintained wrapper library that handles model-specific pooling
internally is a better bet than continuing to patch individual internal
API mismatches one at a time.

Needs sentence-transformers + torch; NOT testable on this development
machine (no torch installed here — same limitation as
generation/generator.py's HFGenerator).
"""

import numpy as np


class CodeSageRetriever:
    # Caché au niveau classe (un seul modèle chargé, réutilisé) : cette
    # classe est instanciée à chaque (dépôt, tâche) dans la boucle de
    # run_benchmark.py — recharger un vrai modèle d'embedding à chaque fois
    # serait inutilement lent.
    _cache: dict[str, object] = {}

    def __init__(self, chunks: list[dict], model_name: str = "codesage/codesage-small-v2"):
        import torch  # noqa: F401 - présence vérifiée tôt, message d'erreur clair sinon
        from sentence_transformers import SentenceTransformer

        if model_name not in CodeSageRetriever._cache:
            CodeSageRetriever._cache[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
        self.model = CodeSageRetriever._cache[model_name]

        self.chunks = chunks
        # header+content, même principe que BM25Retriever : le texte enrichi
        # de cast_scope doit rester disponible au retriever, même si la
        # génération ne le voit que dans une section séparée (voir
        # chunkers/__init__.py).
        texts = [f"{chunk.get('header', '')}\n{chunk['content']}" for chunk in chunks]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True) if texts else np.zeros((0, 1))

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        if len(self.chunks) == 0:
            return []
        query_vec = self.model.encode([query], convert_to_numpy=True)[0]
        chunk_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = chunk_norms * query_norm
        denom[denom == 0] = 1e-9
        scores = (self.embeddings @ query_vec) / denom
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_indices if scores[i] > 0]
