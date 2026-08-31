"""Tests pour chunkers/__init__.py: chunk_file() doit renvoyer la même forme
de dict pour les 3 stratégies, et cast_orig/cast_scope ne doivent différer
QUE dans le texte des en-têtes d'ancêtres (cast_scope), pas dans le fenêtrage
(mêmes frontières de chunk) — sinon la comparaison ne mesurerait plus
seulement l'effet de l'enrichissement scope-aware."""

import unittest

from chunkers import chunk_file


CODE = (
    "class DataProcessor:\n"
    "    def __init__(self):\n"
    "        self.config = {}\n"
    "\n"
    "    def run(self):\n" + "\n".join(f"        x_{i} = {i}" for i in range(60)) + "\n"
)


class TestChunkFileUnifiedInterface(unittest.TestCase):
    def test_empty_code_returns_no_chunks_for_every_strategy(self):
        for strategy in ("fixed", "cast_orig", "cast_scope"):
            self.assertEqual(chunk_file("x.py", "", strategy, max_chunk_size=100), [])

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            chunk_file("x.py", "x = 1", "not_a_real_strategy", max_chunk_size=100)

    def test_every_strategy_returns_the_common_chunk_shape(self):
        for strategy in ("fixed", "cast_orig", "cast_scope"):
            chunks = chunk_file("x.py", CODE, strategy, max_chunk_size=100)
            self.assertGreater(len(chunks), 0, strategy)
            for chunk in chunks:
                self.assertEqual(set(chunk.keys()), {"content", "file_path", "start_line", "end_line"})
                self.assertEqual(chunk["file_path"], "x.py")
                self.assertLessEqual(chunk["start_line"], chunk["end_line"])

    def test_cast_orig_and_cast_scope_have_identical_windowing(self):
        """Même nombre de chunks, mêmes frontières de ligne — seul le texte
        d'en-tête d'ancêtre doit différer entre les deux."""
        orig_chunks = chunk_file("x.py", CODE, "cast_orig", max_chunk_size=100)
        scope_chunks = chunk_file("x.py", CODE, "cast_scope", max_chunk_size=100)
        self.assertEqual(len(orig_chunks), len(scope_chunks))
        for orig, scope in zip(orig_chunks, scope_chunks):
            self.assertEqual(orig["start_line"], scope["start_line"])
            self.assertEqual(orig["end_line"], scope["end_line"])

    def test_cast_scope_header_carries_more_information_than_cast_orig(self):
        orig_chunks = chunk_file("x.py", CODE, "cast_orig", max_chunk_size=100)
        scope_chunks = chunk_file("x.py", CODE, "cast_scope", max_chunk_size=100)
        # le dernier chunk (imbriqué dans run(), sous DataProcessor) doit porter
        # l'annotation d'état seulement côté cast_scope
        self.assertNotIn("State:", orig_chunks[-1]["content"])
        self.assertIn("(State: self.config)", scope_chunks[-1]["content"])

    def test_fixed_strategy_reconstructs_source_across_chunks(self):
        chunks = chunk_file("x.py", CODE, "fixed", max_chunk_size=50)
        self.assertEqual("".join(c["content"] for c in chunks), CODE)


if __name__ == "__main__":
    unittest.main()
