"""Unified chunk_file() entry point for the 3 chunking baselines compared by
run_benchmark.py. Whatever the strategy, a chunk is normalized to the same
shape ({content, header, file_path, start_line, end_line}) so the retriever
and prompt builder downstream never need to know which chunker produced it
— only chunking varies between conditions, everything after it is
identical.

    fixed      -> chunkers.fixed_chunker (Baseline 1: no AST awareness)
    cast_orig  -> astchunk (pip package, unmodified: naive split('\\n')[0] ancestors)
    cast_scope -> astchunk_scope (this project's fork: scope-aware ancestors)

content/header split (NOT astchunk's own chunk_expansion=True mixing them
into one string): a code-completion generator sees "content" as the literal
code it should keep writing in the same style — if ancestor metadata like
"(State: self.db, self.config)" is baked directly into that text (astchunk's
own apply_chunk_expansion() wraps it in a `'''...'''` docstring at the top
of the chunk), the model can imitate that docstring format or otherwise
echo pseudo-metadata instead of real code, which corrupts EM/ES for reasons
that have nothing to do with retrieval or chunking quality. Keeping "header"
separate lets run_benchmark.build_prompt() present it in a clearly-labeled
instruction section instead, and BM25Retriever still indexes header+content
together (see retrieval/bm25_retriever.py) so cast_scope's richer metadata
keeps helping retrieval — only the GENERATION-facing text changes.
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


def _chunk_with_separated_header(builder, code: str, file_path: str) -> list[dict[str, Any]]:
    """Runs the SAME windowing/build_chunk_ancestors() logic as
    builder.chunkify(code, chunk_expansion=True) would, but via the
    lower-level steps directly so we get pure code (ASTChunk.chunk_text,
    never mixed with a header) and the ancestor strings
    (ASTChunk.chunk_ancestors) as two separate values from ONE pass —
    chunkify() only ever returns them pre-merged, and getting them apart by
    calling chunkify() twice (chunk_expansion=False then True) would double
    the actual chunking cost for no reason, since both builders (cast_orig
    and cast_scope) already compute chunk_ancestors as a plain attribute
    before any expansion happens.
    """
    ast = builder.parser.parse(bytes(code, "utf8"))
    ast_windows = list(builder.assign_tree_to_windows(code=code, root_node=ast.root_node))
    ast_windows = builder.add_window_overlapping(ast_windows=ast_windows, chunk_overlap=0)
    ast_chunks = builder.convert_windows_to_chunks(
        ast_windows=ast_windows, repo_level_metadata={"filepath": file_path}, chunk_expansion=False
    )

    chunks = []
    for ast_chunk in ast_chunks:
        chunks.append({
            "content": ast_chunk.chunk_text,
            "header": "\n".join(ast_chunk.chunk_ancestors),
            "file_path": file_path,
            "start_line": ast_chunk.start_line + 1,  # tree-sitter est 0-indexé
            "end_line": ast_chunk.end_line + 1,
        })
    return chunks


def chunk_file(file_path: str, code: str, strategy: str, max_chunk_size: int) -> list[dict[str, Any]]:
    """Chunk one file's source under the given strategy.

    Returns a list of {"content", "header", "file_path", "start_line",
    "end_line"} dicts (1-indexed, inclusive line ranges), regardless of
    strategy. "header" is "" for chunks with no class/function ancestor
    (e.g. "fixed", or top-level module code under cast_orig/cast_scope).
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
                "header": "",
                "file_path": file_path,
                "start_line": line_cursor,
                "end_line": line_cursor + max(n_lines - 1, 0),
            })
            line_cursor += n_lines
        return chunks

    # cast_orig / cast_scope: identical call site, only build_chunk_ancestors()
    # differs, entirely inside the imported library.
    builder = _get_ast_builder(strategy, max_chunk_size)
    try:
        return _chunk_with_separated_header(builder, code, file_path)
    except (SyntaxError, ValueError) as error:
        print(f"Fichier ignoré (échec de parsing, {strategy}) {file_path}: {error}")
        return []
