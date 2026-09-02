"""Tests pour crosscodeeval_adapter.py : résolution owner/repo depuis la
carte des licences, normalisation des tâches CCEval, et surtout
postprocess_completion (portage du _cut_first_statement_completion officiel
de CCEval — logique subtile, jamais testée directement jusqu'ici bien
qu'elle conditionne tout le scoring EM/ES sur CrossCodeEval)."""

import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

from crosscodeeval_adapter import (
    filter_safe_chunks_cceval,
    load_license_map,
    normalize_cceval_record,
    normalize_cceval_tasks,
    postprocess_completion,
    resolve_owner_repo,
)


class TestLoadLicenseMap(unittest.TestCase):
    def _write_map(self, content: str) -> Path:
        f = NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        f.write(content)
        f.close()
        return Path(f.name)

    def test_parses_owner_repo_lines(self):
        path = self._write_map("turboderp/exllama MIT\nhuggingface/diffusers Apache-2.0\n")
        result = load_license_map(path)
        self.assertEqual(result["turboderp-exllama"], "turboderp/exllama")
        self.assertEqual(result["huggingface-diffusers"], "huggingface/diffusers")
        path.unlink()

    def test_skips_blank_lines(self):
        path = self._write_map("turboderp/exllama MIT\n\n\nhuggingface/diffusers Apache-2.0\n")
        result = load_license_map(path)
        self.assertEqual(len(result), 2)
        path.unlink()

    def test_skips_lines_without_slash(self):
        path = self._write_map("not-a-valid-line-at-all\nturboderp/exllama MIT\n")
        result = load_license_map(path)
        self.assertEqual(len(result), 1)
        self.assertIn("turboderp-exllama", result)
        path.unlink()


class TestResolveOwnerRepo(unittest.TestCase):
    def test_resolves_a_matching_prefix(self):
        index = {"turboderp-exllama": "turboderp/exllama"}
        self.assertEqual(resolve_owner_repo("turboderp-exllama-a544085", index), "turboderp/exllama")

    def test_returns_none_when_no_prefix_matches(self):
        index = {"turboderp-exllama": "turboderp/exllama"}
        self.assertIsNone(resolve_owner_repo("someone-else-repo-abc1234", index))

    def test_does_not_match_a_bare_prefix_without_separator(self):
        # "turboderp-exllam" (tronqué) ne doit pas matcher "turboderp-exllama"
        index = {"turboderp-exllama": "turboderp/exllama"}
        self.assertIsNone(resolve_owner_repo("turboderp-exllam-a544085", index))


class TestNormalizeCcevalRecord(unittest.TestCase):
    def _record(self, repository="turboderp-exllama-a544085"):
        return {
            "prompt": "def f():\n    ",
            "groundtruth": "    return 1",
            "metadata": {
                "task_id": "turboderp/exllama/0",
                "repository": repository,
                "file": "model/config.py",
                "context_start_lineno": 3,
                "groundtruth_start_lineno": 4,
            },
        }

    def test_normalizes_a_resolvable_record(self):
        index = {"turboderp-exllama": "turboderp/exllama"}
        task = normalize_cceval_record(self._record(), index)
        self.assertIsNotNone(task)
        self.assertEqual(task["metadata"]["owner_repo"], "turboderp/exllama")
        self.assertEqual(task["metadata"]["ground_truth"], "    return 1")
        self.assertEqual(task["metadata"]["fpath_tuple"], ["turboderp-exllama-a544085", "model", "config.py"])
        self.assertEqual(task["metadata"]["line_no"], 4)

    def test_returns_none_for_unresolvable_repository(self):
        index = {"turboderp-exllama": "turboderp/exllama"}
        task = normalize_cceval_record(self._record(repository="unknown-repo-xyz9999"), index)
        self.assertIsNone(task)

    def test_normalize_tasks_skips_unresolvable_and_keeps_the_rest(self):
        index = {"turboderp-exllama": "turboderp/exllama"}
        records = [self._record(), self._record(repository="unknown-repo-xyz9999")]
        tasks = normalize_cceval_tasks(records, index)
        self.assertEqual(len(tasks), 1)


class TestFilterSafeChunksCceval(unittest.TestCase):
    def test_excludes_the_entire_task_file_not_just_lines_after_the_gap(self):
        chunks = [
            {"file_path": "/repo/model/config.py", "content": "a"},
            {"file_path": "/repo/model/other.py", "content": "b"},
        ]
        safe = filter_safe_chunks_cceval(chunks, Path("/repo"), ["repo-name", "model", "config.py"])
        self.assertEqual(len(safe), 1)
        self.assertEqual(safe[0]["file_path"], "/repo/model/other.py")


class TestPostprocessCompletion(unittest.TestCase):
    """Le prompt est toujours une fonction avec un corps déjà commencé
    (nécessaire pour que le parseur ait un point d'ancrage cohérent) —
    matche la forme réelle des tâches CCEval (complétion à l'intérieur
    d'un corps de fonction/méthode)."""

    PROMPT = "def f():\n    x = 1\n"

    def test_empty_completion_is_returned_unchanged(self):
        self.assertEqual(postprocess_completion(self.PROMPT, ""), "")

    def test_whitespace_only_completion_is_returned_unchanged(self):
        self.assertEqual(postprocess_completion(self.PROMPT, "   "), "   ")

    def test_cuts_after_the_first_complete_statement(self):
        completion = "y = 2\nz = 3\n"
        result = postprocess_completion(self.PROMPT, completion)
        self.assertEqual(result, "y = 2")

    def test_single_complete_statement_followed_by_a_truncated_one(self):
        completion = "y = 2\nz = "
        result = postprocess_completion(self.PROMPT, completion)
        self.assertEqual(result, "y = 2")

    def test_return_statement_is_a_valid_cut_boundary(self):
        completion = "return x + 1\nprint('more')\n"
        result = postprocess_completion(self.PROMPT, completion)
        self.assertEqual(result, "return x + 1")

    def test_docstring_in_prompt_tail_shrinks_the_lookback_window(self):
        # Le prompt se termine par une docstring (triple quotes) -- la
        # boucle de recul (voir postprocess_completion) doit réduire la
        # fenêtre pour ne jamais laisser un ''' ou un """ non refermé dans
        # prompt_tail, sinon le parsing du texte combiné serait faussé.
        prompt = 'def f():\n    """docstring"""\n    x = 1\n'
        completion = "y = 2\nz = 3\n"
        result = postprocess_completion(prompt, completion)
        self.assertEqual(result, "y = 2")

    def test_result_is_never_longer_than_the_raw_completion(self):
        # Propriété générale, quel que soit le contenu exact retourné :
        # postprocess_completion ne doit jamais ALLONGER la complétion.
        completion = "y = 2\nz = 3\nw = 4\n"
        result = postprocess_completion(self.PROMPT, completion)
        self.assertLessEqual(len(result), len(completion))


if __name__ == "__main__":
    unittest.main()
