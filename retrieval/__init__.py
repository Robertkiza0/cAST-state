from .agentic_reranker import AgenticScopeReranker, extract_query_scope_vars
from .bm25_retriever import BM25Retriever
from .codesage_retriever import CodeSageRetriever

RETRIEVERS = ("bm25", "codesage", "agentic")


def get_retriever(name: str, chunks: list[dict]):
    if name == "bm25":
        return BM25Retriever(chunks)
    if name == "codesage":
        return CodeSageRetriever(chunks)
    if name == "agentic":
        return AgenticScopeReranker(chunks)
    raise ValueError(f"Unknown retriever {name!r}, expected one of {RETRIEVERS}")


__all__ = [
    "AgenticScopeReranker",
    "BM25Retriever",
    "CodeSageRetriever",
    "RETRIEVERS",
    "extract_query_scope_vars",
    "get_retriever",
]
