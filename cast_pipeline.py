"""Adapte la vraie librairie cAST (yilinjz/astchunk, papier EMNLP 2025) au
format attendu par weighted_ast_attention_score et par les fonctions communes
du pipeline (filter_safe_chunks, build_prompt, ...) : file_path, line_start,
line_end, raw_code, identifiers, chunk_imports.

Contrairement à container_ast_chunker (fait maison, budget en lignes), on
utilise ici l'algorithme split-then-merge publié : taille en caractères non-
blancs, fusion gloutonne des fenêtres sœurs (ASTChunkBuilder.chunkify). Sert
de 3e stratégie de chunking à comparer à ast_chunker (fenêtres glissantes) et
container_ast_chunker (conteneurs faits maison).
"""

import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Any

from astchunk import ASTChunkBuilder

from container_weighted_pipeline import extract_chunk_identifiers

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PROJECT_DIR / "data" / "cache" / "cast_chunks"

# Valeur retenue par le papier après l'étude de sensibilité (Table 4) : le
# retrieval et la génération culminent entre 2000 et 2500 caractères non-blancs.
DEFAULT_MAX_CHUNK_SIZE = 2000

_builder_cache: dict[int, ASTChunkBuilder] = {}


def _get_builder(max_chunk_size: int) -> ASTChunkBuilder:
    """Un ASTChunkBuilder par taille de chunk (le parser tree-sitter interne
    est coûteux à recréer à chaque appel)."""
    if max_chunk_size not in _builder_cache:
        _builder_cache[max_chunk_size] = ASTChunkBuilder(
            max_chunk_size=max_chunk_size,
            language="python",
            metadata_template="default",
        )
    return _builder_cache[max_chunk_size]


def chunk_file_cast(
    file_path: str, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
) -> list[dict[str, Any]]:
    try:
        code = open(file_path, "r", encoding="utf-8").read()
    except OSError as error:
        print(f"Erreur de lecture de {file_path}: {error}")
        return []

    if not code.strip():
        return []

    builder = _get_builder(max_chunk_size)
    try:
        raw_windows = builder.chunkify(code, repo_level_metadata={"filepath": file_path})
    except (SyntaxError, ValueError) as error:
        print(f"Fichier ignoré (échec de parsing) {file_path}: {error}")
        return []

    chunks = []
    for window in raw_windows:
        code_text = window["content"]
        metadata = window["metadata"]
        symbols, imports = extract_chunk_identifiers(code_text)
        chunks.append({
            "file_path": file_path,
            "line_start": metadata["start_line_no"] + 1,  # tree-sitter est 0-indexé
            "line_end": metadata["end_line_no"] + 1,
            "raw_code": code_text,
            "identifiers": sorted(symbols | imports),
            "chunk_imports": sorted(imports),
        })
    return chunks


def load_and_chunk_repo_cast(
    dir_path: str, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                all_chunks.extend(chunk_file_cast(file_path, max_chunk_size=max_chunk_size))
    return all_chunks


def _repo_fingerprint(dir_path: str) -> str:
    entries = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(root, filename)
                stat = os.stat(file_path)
                entries.append((os.path.relpath(file_path, dir_path), stat.st_mtime_ns, stat.st_size))
    entries.sort()
    return hashlib.sha1(repr(entries).encode("utf-8")).hexdigest()


def load_and_chunk_repo_cast_cached(
    dir_path: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> list[dict[str, Any]]:
    safe_name = re.sub(r"[^\w.-]", "_", os.path.normpath(os.path.abspath(dir_path)))
    cache_file = Path(cache_dir) / f"{safe_name}_size{max_chunk_size}.pkl"
    fingerprint = _repo_fingerprint(dir_path)

    if cache_file.exists():
        with cache_file.open("rb") as file:
            cached = pickle.load(file)
        if cached.get("fingerprint") == fingerprint:
            return cached["chunks"]

    chunks = load_and_chunk_repo_cast(dir_path, max_chunk_size=max_chunk_size)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as file:
        pickle.dump({"fingerprint": fingerprint, "chunks": chunks}, file)
    return chunks


def retrieve_top_k_weighted_cast(
    query_vars: dict[str, float],
    query_imports: set[str],
    chunks: list[dict[str, Any]],
    doc_weights: dict[str, float],
    k: int = 10,
    var_weight: float = 2.0,
    import_weight: float = 2.5,
) -> list[dict[str, Any]]:
    """Comme retrieve_top_k_weighted_container, mais sur des chunks produits
    par le vrai algorithme cAST (split-then-merge, taille en caractères)."""
    from weighted_ast_scorer import weighted_ast_attention_score

    scored = []
    for chunk in chunks:
        chunk_symbols = set(chunk["identifiers"]) - set(chunk["chunk_imports"])
        chunk_imports = set(chunk["chunk_imports"])
        score = weighted_ast_attention_score(
            query_vars, query_imports, chunk_symbols, chunk_imports, doc_weights,
            var_weight=var_weight, import_weight=import_weight,
        )
        scored.append({**chunk, "score": score})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:k]
