import unittest

from retrieval.bm25_retriever import BM25Retriever


def make_chunk(content: str, **extra) -> dict:
    return {"content": content, "file_path": "x.py", "start_line": 1, "end_line": 1, **extra}


class TestBM25Retriever(unittest.TestCase):
    def test_empty_corpus_returns_no_results(self):
        retriever = BM25Retriever([])
        self.assertEqual(retriever.retrieve("anything", k=5), [])

    def test_empty_query_returns_no_results(self):
        retriever = BM25Retriever([make_chunk("def foo(): pass")])
        self.assertEqual(retriever.retrieve("", k=5), [])

    def test_exact_term_match_ranks_above_unrelated_chunks(self):
        # >2 documents on purpose: with exactly 2 docs, a term present in
        # exactly 1 of them hits classic BM25's IDF formula at precisely
        # idf=0 (log((N-n+0.5)/(n+0.5)) with N=2,n=1 -> log(1)=0), which is a
        # degenerate-corpus artifact, not something realistic corpora hit.
        chunks = [
            make_chunk("def unrelated_a(): return 1", id="a"),
            make_chunk("def unrelated_b(): return 2", id="b"),
            make_chunk("def unrelated_c(): return 3", id="c"),
            make_chunk("def compute_stats(values): return sum(values)", id="relevant"),
        ]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("call compute_stats to get the sum", k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "relevant")

    def test_k_limits_number_of_results(self):
        chunks = [make_chunk(f"def foo_{i}(): return foo_shared_term", id=i) for i in range(10)]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("foo_shared_term", k=3)
        self.assertLessEqual(len(results), 3)

    def test_chunk_with_no_term_overlap_is_excluded(self):
        chunks = [make_chunk("import numpy as np", id="numpy_import")]
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("completely_different_identifier_xyz", k=5)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
