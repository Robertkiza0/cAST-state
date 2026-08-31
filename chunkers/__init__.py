"""Unified chunk_file() entry point for the 3 chunking baselines compared by
run_benchmark.py. Whatever the strategy, a chunk is normalized to the same
shape ({content, file_path, start_line, end_line}) so the retriever and
prompt builder downstream never need to know which chunker produced it —
only chunking varies between conditions, everything after it is identical.

    fixed      -> chunkers.fixed_chunker (Baseline 1: no AST awareness)
    cast_orig  -> astchunk (pip package, unmodified: naive split('\\n')[0] ancestors)
    cast_scope -> astchunk_scope (this project's fork: scope-aware ancestors)
"""

from typing import Any

from .fixed_chunker import fixed_size_chunk

STRATEGIES = ("fixed", "cast_orig", "cast_scope")

_builder_cache: dict[tuple[str, int], Any] = {}


def _get_ast_builder(strategy: str, max_chunk_size: int):
    cache_key = (strategy, max_chunk_size)
    if cache_key in _builder_cache:
        return _builder_cache[cache_key]

    if strategy == "cast_orig":
        from astchunk import ASTChunkBuilder  # pip-installed, unmodified upstream
    elif strategy == "cast_scope":
        from astchunk_scope import ASTChunkBuilder  # this project's scope-aware fork
    else:
        raise ValueError(f"Not an AST-based strategy: {strategy!r}")

    builder = ASTChunkBuilder(max_chunk_size=max_chunk_size, language="python", metadata_template="default")
    _builder_cache[cache_key] = builder
    return builder


def chunk_file(file_path: str, code: str, strategy: str, max_chunk_size: int) -> list[dict[str, Any]]:
    """Chunk one file's source under the given strategy.

    Returns a list of {"content", "file_path", "start_line", "end_line"}
    dicts (1-indexed, inclusive line ranges), regardless of strategy.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown chunking strategy {strategy!r}, expected one of {STRATEGIES}")

    if not code.strip():
        return []

    if strategy == "fixed":
        chunks = []
        line_cursor = 1
        for piece in fixed_size_chunk(code, max_chunk_size):
            n_lines = len(piece.splitlines())
            chunks.append({
                "content": piece,
                "file_path": file_path,
                "start_line": line_cursor,
                "end_line": line_cursor + max(n_lines - 1, 0),
            })
            line_cursor += n_lines
        return chunks

    # cast_orig / cast_scope: both go through ASTChunkBuilder.chunkify with
    # chunk_expansion=True — the only difference between the two conditions
    # is which build_chunk_ancestors() implementation gets called, entirely
    # inside the imported library; this call site is identical for both.
    builder = _get_ast_builder(strategy, max_chunk_size)
    try:
        windows = builder.chunkify(
            code, repo_level_metadata={"filepath": file_path}, chunk_expansion=True
        )
    except (SyntaxError, ValueError) as error:
        print(f"Fichier ignoré (échec de parsing, {strategy}) {file_path}: {error}")
        return []

    chunks = []
    for window in windows:
        metadata = window["metadata"]
        chunks.append({
            "content": window["content"],
            "file_path": file_path,
            "start_line": metadata["start_line_no"] + 1,  # tree-sitter est 0-indexé
            "end_line": metadata["end_line_no"] + 1,
        })
    return chunks
