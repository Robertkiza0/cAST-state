"""Tests pour retrieval/agentic_reranker.py — la partie testable sans torch
(pas de CodeSage réel ici) : extract_query_scope_vars (extraction pure), et
la logique de reranking elle-même via un faux retriever dense injecté
(dense_retriever=...), qui isole le SCORE de reranking de tout ce qui
dépend d'un vrai modèle d'embedding."""

import unittest

from retrieval.agentic_reranker import AgenticScopeReranker, extract_query_scope_vars


def make_chunk(header: str = "", content: str = "code") -> dict:
    return {"header": header, "content": content, "file_path": "x.py", "start_line": 1, "end_line": 1}


class FakeDenseRetriever:
    """Renvoie toujours les mêmes candidats dans le même ordre, quelle que
    soit la requête — isole le reranking du calcul d'embedding réel."""

    def __init__(self, candidates: list[dict]):
        self.candidates = candidates

    def retrieve(self, query: str, k: int = 20) -> list[dict]:
        return self.candidates[:k]


class TestExtractQueryScopeVars(unittest.TestCase):
    def test_extracts_self_attributes_mentioned_in_the_tail(self):
        code = "def run(self):\n    x = self.db.execute()\n    y = self.config['a']\n"
        result = extract_query_scope_vars(code)
        self.assertEqual(result, ["config", "db"])

    def test_deduplicates_repeated_mentions(self):
        code = "self.db.a()\nself.db.b()\n"
        result = extract_query_scope_vars(code)
        self.assertEqual(result, ["db"])

    def test_no_self_mentions_returns_empty_list(self):
        code = "x = compute(1, 2)\ny = x + 1\n"
        self.assertEqual(extract_query_scope_vars(code), [])

    def test_only_looks_at_the_tail_not_the_whole_code(self):
        code = "self.only_in_head = 1\n" + "\n".join(f"pad_{i} = {i}" for i in range(30))
        result = extract_query_scope_vars(code, tail_lines=5)
        self.assertNotIn("only_in_head", result)


class TestAgenticScopeReranker(unittest.TestCase):
    def test_chunk_matching_a_query_scope_var_is_boosted_above_a_plain_one(self):
        matching = make_chunk(header="class Foo: (State: self.db)")
        plain = make_chunk(header="class Bar:")
        # le dense retriever (factice) renvoie 'plain' en premier -- le
        # reranking doit quand meme faire remonter 'matching' en tete,
        # puisque la requete mentionne self.db.
        dense = FakeDenseRetriever([plain, matching])
        reranker = AgenticScopeReranker([], dense_retriever=dense)

        query = "def run(self):\n    return self.db.execute()\n"
        results = reranker.retrieve(query, k=2)
        self.assertEqual(results[0], matching)

    def test_decorator_present_gets_a_small_bonus(self):
        decorated = make_chunk(header="@app.get('/x') def route():")
        plain = make_chunk(header="def other():")
        dense = FakeDenseRetriever([plain, decorated])
        reranker = AgenticScopeReranker([], dense_retriever=dense)

        results = reranker.retrieve("no self mentions here", k=2)
        self.assertEqual(results[0], decorated)

    def test_no_query_scope_vars_preserves_dense_order(self):
        first = make_chunk(header="class A:")
        second = make_chunk(header="class B:")
        dense = FakeDenseRetriever([first, second])
        reranker = AgenticScopeReranker([], dense_retriever=dense)

        results = reranker.retrieve("no self mentions, no decorators", k=2)
        self.assertEqual(results, [first, second])

    def test_k_limits_the_number_of_results(self):
        candidates = [make_chunk(header=f"class C{i}:") for i in range(10)]
        dense = FakeDenseRetriever(candidates)
        reranker = AgenticScopeReranker([], dense_retriever=dense)
        results = reranker.retrieve("query", k=3)
        self.assertEqual(len(results), 3)

    def test_candidate_pool_is_passed_to_the_dense_retriever(self):
        seen_k = {}

        class RecordingDense:
            def retrieve(self, query, k=20):
                seen_k["k"] = k
                return []

        reranker = AgenticScopeReranker([], candidate_pool=7, dense_retriever=RecordingDense())
        reranker.retrieve("query", k=3)
        self.assertEqual(seen_k["k"], 7)


if __name__ == "__main__":
    unittest.main()
