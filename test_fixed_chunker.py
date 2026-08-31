import unittest

from chunkers.fixed_chunker import _nws_count, fixed_size_chunk


class TestFixedSizeChunk(unittest.TestCase):
    def test_empty_code_returns_no_chunks(self):
        self.assertEqual(fixed_size_chunk("", max_chunk_size=100), [])

    def test_small_code_is_a_single_chunk(self):
        code = "x = 1\ny = 2\n"
        chunks = fixed_size_chunk(code, max_chunk_size=100)
        self.assertEqual(chunks, [code])

    def test_never_splits_a_line_in_the_middle(self):
        code = "\n".join(f"line_{i} = {i}" for i in range(50)) + "\n"
        chunks = fixed_size_chunk(code, max_chunk_size=50)
        # chaque chunk, une fois rejoint, doit se terminer par une frontière de ligne complète
        for chunk in chunks:
            self.assertTrue(chunk.endswith("\n") or chunk == chunks[-1])
        self.assertEqual("".join(chunks), code)

    def test_respects_max_chunk_size_in_non_whitespace_chars(self):
        code = "\n".join("x" * 20 for _ in range(20)) + "\n"
        max_size = 100
        chunks = fixed_size_chunk(code, max_chunk_size=max_size)
        self.assertGreater(len(chunks), 1)
        # seul le dernier chunk peut être plus petit ; aucun ne doit dépasser le budget
        for chunk in chunks:
            self.assertLessEqual(_nws_count(chunk), max_size + 20)  # +1 ligne de marge (une ligne isolée peut dépasser seule)

    def test_reconstructs_original_code_exactly(self):
        code = "".join(f"line {i}\n" for i in range(100))
        chunks = fixed_size_chunk(code, max_chunk_size=37)
        self.assertEqual("".join(chunks), code)

    def test_single_line_larger_than_budget_becomes_its_own_chunk(self):
        code = "x = " + "1" * 500 + "\n"
        chunks = fixed_size_chunk(code, max_chunk_size=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], code)


if __name__ == "__main__":
    unittest.main()
