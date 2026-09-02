"""Tests pour experiments/mcnemar_significance.py — nouveau, jamais testé
jusqu'ici, alors que c'est précisément le script qui décide si un écart EM
brut entre stratégies est réel ou du bruit (voir README/mémoire projet)."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

from mcnemar_significance import load_results, mcnemar_test  # noqa: E402


def write_jsonl(records: list[dict]) -> Path:
    f = NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for record in records:
        f.write(json.dumps(record) + "\n")
    f.close()
    return Path(f.name)


class TestLoadResults(unittest.TestCase):
    def test_groups_by_dataset_then_strategy_then_task_id(self):
        path = write_jsonl([
            {"dataset": "repoeval", "task_id": "t1", "strategy": "fixed", "em": 1, "es": 0.9, "pass1": 1, "prediction": "x"},
            {"dataset": "repoeval", "task_id": "t1", "strategy": "cast_orig", "em": 0, "es": 0.5, "pass1": 0, "prediction": "y"},
            {"dataset": "cceval", "task_id": "t2", "strategy": "fixed", "em": 0, "es": 0.1, "pass1": 0, "prediction": "z"},
        ])
        results = load_results(path)
        self.assertEqual(set(results.keys()), {"repoeval", "cceval"})
        self.assertEqual(results["repoeval"]["fixed"]["t1"], 1)
        self.assertEqual(results["repoeval"]["cast_orig"]["t1"], 0)
        self.assertEqual(results["cceval"]["fixed"]["t2"], 0)
        path.unlink()

    def test_skips_blank_lines(self):
        f = NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
        f.write('{"dataset": "repoeval", "task_id": "t1", "strategy": "fixed", "em": 1, "es": 0.9, "pass1": 1, "prediction": "x"}\n')
        f.write("\n")
        f.write("   \n")
        f.close()
        path = Path(f.name)
        results = load_results(path)
        self.assertEqual(len(results["repoeval"]["fixed"]), 1)
        path.unlink()


class TestMcnemarTest(unittest.TestCase):
    def test_no_discordant_pairs_returns_p_one(self):
        em_a = {"t1": 1, "t2": 0, "t3": 1}
        em_b = {"t1": 1, "t2": 0, "t3": 1}
        p = mcnemar_test("a", "b", em_a, em_b)
        self.assertEqual(p, 1.0)

    def test_perfectly_balanced_discordant_pairs_is_not_significant(self):
        # 5 tâches où seul A a raison, 5 où seul B a raison -> parfaitement
        # symétrique, ne doit jamais être significatif.
        em_a = {f"a{i}": 1 for i in range(5)} | {f"b{i}": 0 for i in range(5)}
        em_b = {f"a{i}": 0 for i in range(5)} | {f"b{i}": 1 for i in range(5)}
        p = mcnemar_test("a", "b", em_a, em_b)
        self.assertGreaterEqual(p, 0.05)

    def test_heavily_skewed_discordant_pairs_is_significant(self):
        # 20 tâches où seul A a raison, aucune où B a raison seul -> devrait
        # ressortir significatif (asymétrie extrême).
        em_a = {f"t{i}": 1 for i in range(20)}
        em_b = {f"t{i}": 0 for i in range(20)}
        p = mcnemar_test("a", "b", em_a, em_b)
        self.assertLess(p, 0.05)

    def test_only_shared_task_ids_are_compared(self):
        em_a = {"t1": 1, "t2": 0, "only_in_a": 1}
        em_b = {"t1": 1, "t2": 0, "only_in_b": 0}
        # aucune paire discordante parmi les tâches PARTAGÉES (t1, t2) ->
        # p=1.0, peu importe les tâches non partagées.
        p = mcnemar_test("a", "b", em_a, em_b)
        self.assertEqual(p, 1.0)


if __name__ == "__main__":
    unittest.main()
