"""
astchunk_scope - cAST-Scope: a scope-aware fork of astchunk (yilinjz/astchunk,
Zhang et al., "cAST: Enhancing Code Retrieval-Augmented Generation with
Structural Chunking via Abstract Syntax Tree", EMNLP 2025 Findings).

The behavioral change from the original is in ASTChunk.build_chunk_ancestors()
(see astchunk.py): class ancestors are annotated with their self.* instance
state, and function ancestors are prefixed with their decorators. Windowing,
merging, overlap, and the size metric are untouched. astchunk_builder.py has
one small addition: a per-chunkify() cache dict threaded into each ASTChunk
so a class/function ancestor's annotation is computed once and reused across
every chunk nested under it, not recomputed per chunk (see ASTChunk.__init__
and build_chunk_ancestors) — this keeps annotation cost negligible even for
large classes with many chunks. See LICENSE for the original MIT license
this fork is derived under.
"""

from .astchunk_builder import ASTChunkBuilder
from .astchunk import ASTChunk
from .astnode import ASTNode
from .preprocessing import (
    ByteRange,
    IntRange,
    preprocess_nws_count,
    get_nws_count,
    get_nws_count_direct,
    get_nodes_in_brange,
    get_largest_node_in_brange
)

__version__ = "0.1.0-scope"

__all__ = [
    "ASTChunkBuilder",
    "ASTChunk",
    "ASTNode",
    "ByteRange",
    "IntRange",
    "preprocess_nws_count",
    "get_nws_count",
    "get_nws_count_direct",
    "get_nodes_in_brange",
    "get_largest_node_in_brange"
]
