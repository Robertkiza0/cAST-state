"""Tests pour la mitigation du risque d'hallucination sur les métadonnées
de chunk (voir run_benchmark.py: format_chunk_block, build_prompt,
sanitize_generated_code, STOP_SEQUENCES).

Motivation : coller la métadonnée d'ancêtre (ex. "(State: self.db)")
directement dans le texte de code envoyé au générateur — ce que fait
astchunk.apply_chunk_expansion() par défaut, via un faux docstring
''' ... ''' au début du chunk — risque de faire imiter ce format par le
modèle, ou halluciner du pseudo-code, plutôt que de continuer en Python
valide. Trois verrous : (1) séparer la métadonnée dans une section
d'instruction distincte du code, jamais mélangée dedans ; (2) des
séquences d'arrêt côté génération ; (3) un nettoyage de secours côté
post-traitement avant le scoring.
"""

import unittest

from run_benchmark import (
    CODE_SNIPPET_MARKER,
    CONTEXT_INSTRUCTION_MARKER,
    STOP_SEQUENCES,
    build_prompt,
    format_chunk_block,
    sanitize_generated_code,
)


def make_chunk(content: str, header: str = "", file_path: str = "x.py") -> dict:
    return {"content": content, "header": header, "file_path": file_path, "start_line": 1, "end_line": 1}


class TestFormatChunkBlock(unittest.TestCase):
    def test_header_and_content_are_never_mixed_in_the_same_line(self):
        chunk = make_chunk("def foo():\n    return 1", header="class Foo: (State: self.x)")
        block = format_chunk_block(chunk)
        # le code doit apparaitre exactement tel quel, sans que le header
        # soit injecte au milieu ou autour de lui
        self.assertIn("def foo():\n    return 1", block)

    def test_header_appears_in_a_labeled_section_before_the_code(self):
        chunk = make_chunk("def foo(): pass", header="class Foo: (State: self.x)")
        block = format_chunk_block(chunk)
        header_pos = block.index("Scope: class Foo: (State: self.x)")
        code_pos = block.index(CODE_SNIPPET_MARKER)
        self.assertLess(header_pos, code_pos, "le header doit venir avant la section code")

    def test_chunk_without_header_has_no_scope_line(self):
        chunk = make_chunk("x = 1", header="")
        block = format_chunk_block(chunk)
        self.assertNotIn("Scope:", block)

    def test_file_path_always_present(self):
        chunk = make_chunk("x = 1", file_path="src/module.py")
        block = format_chunk_block(chunk)
        self.assertIn("File: src/module.py", block)


class TestBuildPromptSeparation(unittest.TestCase):
    def test_prompt_never_mixes_scope_metadata_into_a_code_line(self):
        chunks = [make_chunk("self.db.execute(x)", header="class Service: (State: self.db, self.logger)")]
        prompt = build_prompt("def run(self):\n    ", chunks, max_context_chars=3000)
        # "State:" doit apparaitre seulement sur une ligne "Scope:", jamais
        # accolee au code sur la meme ligne qu'une instruction Python
        for line in prompt.split("\n"):
            if "State:" in line:
                self.assertTrue(line.strip().startswith("Scope:"), line)

    def test_prompt_contains_both_marker_sections_when_chunks_retrieved(self):
        chunks = [make_chunk("x = 1", header="class A:")]
        prompt = build_prompt("y = 2", chunks, max_context_chars=3000)
        self.assertIn(CONTEXT_INSTRUCTION_MARKER, prompt)
        self.assertIn(CODE_SNIPPET_MARKER, prompt)

    def test_empty_retrieval_falls_back_to_just_unfinished_code(self):
        prompt = build_prompt("y = 2", [], max_context_chars=3000)
        self.assertEqual(prompt, "y = 2")
        self.assertNotIn(CONTEXT_INSTRUCTION_MARKER, prompt)


class TestSanitizeGeneratedCode(unittest.TestCase):
    def test_removes_hallucinated_context_instruction_line(self):
        prediction = f"{CONTEXT_INSTRUCTION_MARKER}\nFile: fake.py\nreturn x + 1"
        cleaned = sanitize_generated_code(prediction)
        self.assertNotIn(CONTEXT_INSTRUCTION_MARKER, cleaned)
        self.assertNotIn("File: fake.py", cleaned)
        self.assertIn("return x + 1", cleaned)

    def test_removes_hallucinated_scope_line(self):
        prediction = "Scope: class Fake: (State: self.y)\nreturn x + 1"
        cleaned = sanitize_generated_code(prediction)
        self.assertNotIn("Scope:", cleaned)
        self.assertIn("return x + 1", cleaned)

    def test_normal_code_without_any_marker_is_untouched(self):
        prediction = "def add(a, b):\n    return a + b"
        self.assertEqual(sanitize_generated_code(prediction), prediction)

    def test_does_not_strip_legitimate_code_containing_the_word_scope(self):
        # "Scope:" avec le deux-points est le marqueur ; une simple mention
        # du mot "scope" dans du vrai code ne doit pas être touchée.
        prediction = "scope_value = get_scope()\nreturn scope_value"
        self.assertEqual(sanitize_generated_code(prediction), prediction)


class TestStopSequencesCoverMarkers(unittest.TestCase):
    def test_stop_sequences_include_both_markers(self):
        joined = " ".join(STOP_SEQUENCES)
        self.assertIn(CONTEXT_INSTRUCTION_MARKER, joined)
        self.assertIn(CODE_SNIPPET_MARKER, joined)


if __name__ == "__main__":
    unittest.main()
