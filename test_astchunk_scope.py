"""Tests pour l'extension cAST-Scope de build_chunk_ancestors (astchunk_scope/astchunk.py) :
- un ancêtre class_definition est annoté avec son état self.* (Limitation #1 de cAST)
- un ancêtre function_definition est préfixé de ses décorateurs
- aucune régression sur le comportement non modifié (fenêtrage, taille, texte reconstruit)
"""

import time
import unittest

from astchunk_scope import ASTChunkBuilder


def chunkify(code: str, max_chunk_size: int, **configs) -> list[dict]:
    builder = ASTChunkBuilder(max_chunk_size=max_chunk_size, language="python", metadata_template="default")
    return builder.chunkify(code, chunk_expansion=True, **configs)


class TestClassStateAnnotation(unittest.TestCase):
    def test_simple_attributes_are_listed_sorted(self):
        code = (
            "class DataProcessor:\n"
            "    def __init__(self):\n"
            "        self.logger = get_logger()\n"
            "        self.config = {}\n"
            "        self.db = None\n"
            "\n"
            "    def run(self):\n" + "\n".join(f"        x_{i} = {i}" for i in range(80)) + "\n"
        )
        chunks = chunkify(code, max_chunk_size=100)
        self.assertGreater(len(chunks), 1, "le test suppose un découpage en plusieurs chunks")
        last_content = chunks[-1]["content"]
        self.assertIn("class DataProcessor:", last_content)
        self.assertIn("(State: self.config, self.db, self.logger)", last_content)

    def test_tuple_unpacking_attributes_are_captured(self):
        code = (
            "class Point:\n"
            "    def __init__(self):\n"
            "        self.x, self.y = 0, 0\n"
            "\n"
            "    def reset(self):\n" + "\n".join(f"        pad_{i} = {i}" for i in range(80)) + "\n"
        )
        chunks = chunkify(code, max_chunk_size=100)
        last_content = chunks[-1]["content"]
        self.assertIn("self.x", last_content)
        self.assertIn("self.y", last_content)

    def test_augmented_assignment_attribute_is_captured(self):
        code = (
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.count = 0\n"
            "\n"
            "    def increment(self):\n"
            "        self.count += 1\n"
            "        return self.count\n"
            "\n"
            "    def other(self):\n" + "\n".join(f"        pad_{i} = {i}" for i in range(80)) + "\n"
        )
        chunks = chunkify(code, max_chunk_size=100)
        last_content = chunks[-1]["content"]
        self.assertIn("(State: self.count)", last_content)

    def test_no_self_attributes_means_no_state_annotation(self):
        code = (
            "class Stateless:\n"
            "    def compute(self, x):\n"
            "        return x * 2\n"
            "\n"
            "    def other(self, x):\n" + "\n".join(f"        pad_{i} = {i}" for i in range(80)) + "\n"
        )
        chunks = chunkify(code, max_chunk_size=100)
        last_content = chunks[-1]["content"]
        self.assertIn("class Stateless:", last_content)
        self.assertNotIn("(State:", last_content)

    def test_nested_class_attributes_are_not_leaked_to_outer_class(self):
        code = (
            "class Outer:\n"
            "    def __init__(self):\n"
            "        self.outer_attr = 1\n"
            "\n"
            "    class Inner:\n"
            "        def __init__(self):\n"
            "            self.inner_attr = 2\n"
            "\n"
            "    def run(self):\n" + "\n".join(f"        pad_{i} = {i}" for i in range(80)) + "\n"
        )
        chunks = chunkify(code, max_chunk_size=100)
        outer_state_line = next(line for chunk in chunks for line in chunk["content"].split("\n") if line.strip().startswith("class Outer:"))
        self.assertIn("self.outer_attr", outer_state_line)
        self.assertNotIn("inner_attr", outer_state_line)


class TestFunctionDecoratorAnnotation(unittest.TestCase):
    def test_single_decorator_is_prefixed_to_ancestor_header(self):
        code = (
            "@app.get('/x')\n"
            "def route():\n"
            "    def inner():\n" + "\n".join(f"        pad_{i} = {i}" for i in range(80)) + "\n"
            "    return inner\n"
        )
        chunks = chunkify(code, max_chunk_size=100)
        last_content = chunks[-1]["content"]
        self.assertIn("@app.get('/x') def route():", last_content)

    def test_multiple_decorators_are_all_kept_in_order(self):
        code = (
            "@first\n"
            "@second\n"
            "def route():\n"
            "    def inner():\n" + "\n".join(f"        pad_{i} = {i}" for i in range(80)) + "\n"
            "    return inner\n"
        )
        chunks = chunkify(code, max_chunk_size=100)
        last_content = chunks[-1]["content"]
        self.assertIn("@first @second def route():", last_content)

    def test_undecorated_function_ancestor_has_plain_header(self):
        code = (
            "def route():\n"
            "    def inner():\n" + "\n".join(f"        pad_{i} = {i}" for i in range(80)) + "\n"
            "    return inner\n"
        )
        chunks = chunkify(code, max_chunk_size=100)
        last_content = chunks[-1]["content"]
        self.assertIn("def route():", last_content)
        self.assertNotIn("@", last_content.split("'''")[1])  # dans le bloc d'ancêtres uniquement


class TestNoRegressionOnUnrelatedBehavior(unittest.TestCase):
    def test_chunk_text_reconstructs_source_verbatim_without_expansion(self):
        code = "def f(x):\n    return x + 1\n\ndef g(y):\n    return y - 1\n"
        builder = ASTChunkBuilder(max_chunk_size=2000, language="python", metadata_template="default")
        chunks = builder.chunkify(code)  # chunk_expansion=False par défaut
        self.assertEqual("".join(c["content"] for c in chunks), code)

    def test_small_file_yields_a_single_chunk(self):
        code = "x = 1\ny = 2\n"
        builder = ASTChunkBuilder(max_chunk_size=2000, language="python", metadata_template="default")
        chunks = builder.chunkify(code)
        self.assertEqual(len(chunks), 1)


class TestPerformance(unittest.TestCase):
    def test_ancestor_annotation_amortized_cost_stays_under_one_millisecond_per_chunk(self):
        """Exigence du projet : l'enrichissement ne doit pas mesurablement
        ralentir le chunking (< 1 ms par chunk produit).

        Le cache par appel à chunkify() (voir ASTChunkBuilder.convert_windows_to_chunks)
        fait que seul le PREMIER chunk d'une classe paie le parcours complet de
        son corps pour en extraire l'état self.* ; les chunks suivants de la
        même classe réutilisent le résultat. On mesure donc le coût amorti sur
        tout un appel chunkify() réaliste (20 attributs, une vingtaine de
        chunks), pas un appel isolé à froid — sans cache, un seul chunk isolé
        recalculerait tout à chaque fois, ce qui redeviendrait coûteux à mesure
        que le nombre de chunks nichés dans une même classe grandit.
        """
        code = (
            "class RealisticService:\n"
            "    def __init__(self):\n"
            + "\n".join(f"        self.attr_{i} = {i}" for i in range(20)) + "\n"
            "\n"
            "    def run(self):\n" + "\n".join(f"        pad_{i} = {i}" for i in range(400)) + "\n"
        )
        builder = ASTChunkBuilder(max_chunk_size=100, language="python", metadata_template="default")

        start = time.perf_counter()
        chunks = builder.chunkify(code, chunk_expansion=True)
        elapsed = time.perf_counter() - start

        self.assertGreater(len(chunks), 10, "le test suppose plusieurs chunks nichés dans la même classe")
        avg_ms_per_chunk = (elapsed / len(chunks)) * 1000
        self.assertLess(
            avg_ms_per_chunk, 1.0,
            f"coût amorti trop élevé: {avg_ms_per_chunk:.4f} ms/chunk sur {len(chunks)} chunks",
        )


if __name__ == "__main__":
    unittest.main()
