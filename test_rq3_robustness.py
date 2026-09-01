"""RQ3 (Robustness to Structural Errors): does cAST-Scope's scope expansion
(self.* state extraction, decorator lookup) preserve class context
integrity and structural stability when the code near the cursor is
incomplete or has a syntax error — exactly the situation code completion
operates in (the file being completed is, by construction, not yet
syntactically complete at the cursor position)?

Unlike Python's own `ast` module (raises SyntaxError on any malformed
input — all-or-nothing), tree-sitter is error-tolerant: it produces ERROR
nodes for the malformed region but keeps parsing the rest of the file as
normal nodes. This is exactly why cAST/cAST-Scope can use tree-sitter for
repository-level completion in the first place — cast_orig already relies
on this tolerance for its own (unmodified) chunking. What this file tests
specifically is whether cAST-Scope's ADDED code (_extract_self_attributes,
_decorator_lines — new tree-walking logic cast_orig never had) introduces
NEW crash modes on malformed/incomplete trees that cast_orig wouldn't hit,
and whether the self.*/decorator info it does manage to extract from the
still-valid portion of a broken class is correct rather than corrupted.
"""

import unittest

from astchunk_scope import ASTChunkBuilder


def chunk(code: str, max_chunk_size: int = 50) -> list[dict]:
    """Petit max_chunk_size pour forcer le découpage même sur des snippets
    courts, donc pour exercer build_chunk_ancestors sur un ancêtre réel."""
    builder = ASTChunkBuilder(max_chunk_size=max_chunk_size, language="python", metadata_template="default")
    return builder.chunkify(code, chunk_expansion=True)


class TestTruncatedCodeNearCursor(unittest.TestCase):
    """Simule la situation réelle de complétion : le fichier s'arrête net
    au milieu de quelque chose, parce que c'est exactement là qu'est le
    curseur."""

    def test_truncated_mid_assignment_does_not_crash(self):
        code = (
            "class DataProcessor:\n"
            "    def __init__(self):\n"
            "        self.config = {}\n"
            "        self.db = connect_to_datab"  # coupé en plein milieu du nom
        )
        chunks = chunk(code)  # ne doit pas lever d'exception
        self.assertGreater(len(chunks), 0)

    def test_self_attributes_before_truncation_point_are_still_found(self):
        code = (
            "class DataProcessor:\n"
            "    def __init__(self):\n"
            "        self.config = {}\n"
            "        self.logger = get_logger()\n"
            "        self.db = connect_to_datab"  # coupé, self.db incomplet
            + "\n".join(f"        pad_{i} = {i}" for i in range(30))
        )
        chunks = chunk(code)
        last_content = chunks[-1]["content"]
        # les deux attributs COMPLETS avant la coupure doivent être présents
        self.assertIn("self.config", last_content)
        self.assertIn("self.logger", last_content)

    def test_truncated_right_after_class_header_does_not_crash(self):
        code = "class DataProcessor:\n    def __init__(self):\n        self."
        chunks = chunk(code)
        self.assertGreaterEqual(len(chunks), 0)  # ne doit pas lever, quel que soit le résultat

    def test_decorator_with_truncated_function_does_not_crash(self):
        code = (
            "@app.get('/x')\n"
            "def rou"  # coupé en plein milieu du nom de fonction
        )
        chunks = chunk(code)
        self.assertGreaterEqual(len(chunks), 0)


class TestSyntaxErrorsNotAtEndOfFile(unittest.TestCase):
    """Une vraie erreur de syntaxe (pas juste une coupure en fin de
    fichier) au milieu d'une classe — parenthèse non fermée, etc."""

    def test_unclosed_paren_in_one_method_does_not_break_other_methods(self):
        code = (
            "class Service:\n"
            "    def __init__(self):\n"
            "        self.name = 'svc'\n"
            "\n"
            "    def broken(self):\n"
            "        return compute(1, 2\n"  # parenthèse jamais fermée
            "\n"
            "    def run(self):\n" + "\n".join(f"        pad_{i} = {i}" for i in range(30)) + "\n"
        )
        chunks = chunk(code)  # ne doit pas lever, malgré l'erreur de syntaxe
        self.assertGreater(len(chunks), 0)
        last_content = chunks[-1]["content"]
        self.assertIn("self.name", last_content)

    def test_missing_colon_does_not_crash(self):
        code = (
            "class Broken\n"  # deux-points manquant après le nom de classe
            "    def __init__(self):\n"
            "        self.x = 1\n"
        )
        chunks = chunk(code)
        self.assertGreaterEqual(len(chunks), 0)


class TestEmptyOrDegenerateClasses(unittest.TestCase):
    def test_empty_class_body_does_not_crash(self):
        code = "class Empty:\n    pass\n"
        chunks = chunk(code)
        self.assertGreater(len(chunks), 0)
        self.assertNotIn("(State:", chunks[0]["content"])

    def test_class_with_only_a_docstring_does_not_crash(self):
        code = "class Documented:\n    '''Just a docstring.'''\n"
        chunks = chunk(code)
        self.assertGreater(len(chunks), 0)


class TestParityWithBaselineOnMalformedInput(unittest.TestCase):
    """Sur une entrée cassée, cast_scope ne doit JAMAIS crasher là où
    cast_orig (non modifié) ne crashe pas — la robustesse ajoutée ne doit
    jamais être une régression par rapport à la baseline officielle."""

    MALFORMED_SNIPPETS = [
        "class A:\n    def __init__(self):\n        self.x = ",
        "class B:\n    def m(self):\n        return foo(1, 2\n    def n(self):\n        pass\n",
        "@dec\n",
        "class C\n    def __init__(self):\n        self.y = 1\n",
        "def f(:\n    pass\n",
    ]

    def test_cast_scope_never_crashes_where_cast_orig_does_not(self):
        from astchunk import ASTChunkBuilder as OrigBuilder

        orig_builder = OrigBuilder(max_chunk_size=50, language="python", metadata_template="default")
        scope_builder = ASTChunkBuilder(max_chunk_size=50, language="python", metadata_template="default")

        for snippet in self.MALFORMED_SNIPPETS:
            orig_crashed = False
            try:
                orig_builder.chunkify(snippet, chunk_expansion=True)
            except (SyntaxError, ValueError):
                orig_crashed = True

            if orig_crashed:
                continue  # cast_orig echoue deja sur cette entree, rien a comparer

            try:
                scope_builder.chunkify(snippet, chunk_expansion=True)
            except Exception as error:  # noqa: BLE001 - on veut voir TOUT crash ici
                self.fail(f"cast_scope a crashé là où cast_orig réussit, sur {snippet!r}: {error}")


if __name__ == "__main__":
    unittest.main()
