"""Paired McNemar significance test on run_benchmark.py's saved results
(results/run_*.jsonl), same methodology already used in this project's
earlier work (repocoder-mine) before trusting any raw EM delta between
conditions: fixed vs cast_orig, cast_orig vs cast_scope, fixed vs
cast_scope, per dataset.

Why this matters here specifically: RQ2's raw numbers can look like a real
effect (a few EM points apart) while being fully consistent with noise at
n~300 — this project has hit that exact trap before on a different axis
(scoring signal) and only found out via McNemar. Don't report a RQ2 win or
loss for cast_scope without running this first.

Usage:
    python experiments/mcnemar_significance.py results/run_20260902_120000.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import binomtest
except ImportError:
    print("scipy est requis pour ce script (pip install scipy).", file=sys.stderr)
    raise


def load_results(path: Path) -> dict[str, dict[str, dict[str, int]]]:
    """{dataset: {strategy: {task_id: em}}} à partir du jsonl de run_benchmark.py."""
    results: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            results[record["dataset"]][record["strategy"]][record["task_id"]] = record["em"]
    return results


def mcnemar_test(name_a: str, name_b: str, em_a: dict[str, int], em_b: dict[str, int]) -> float:
    shared_ids = sorted(set(em_a) & set(em_b))
    b = sum(1 for tid in shared_ids if em_a[tid] == 1 and em_b[tid] == 0)
    c = sum(1 for tid in shared_ids if em_a[tid] == 0 and em_b[tid] == 1)
    n_discordant = b + c
    if n_discordant == 0:
        print(f"  {name_a} vs {name_b}: {len(shared_ids)} tâches communes, aucune paire discordante -> p=1.0")
        return 1.0
    p_value = binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue
    flag = "  *** significatif (p<0.05)" if p_value < 0.05 else ""
    print(
        f"  {name_a} vs {name_b}: {len(shared_ids)} tâches communes | "
        f"{name_a} seul correct={b}  {name_b} seul correct={c}  p={p_value:.4f}{flag}"
    )
    return p_value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_path", help="Fichier results/run_*.jsonl produit par run_benchmark.py")
    args = parser.parse_args()

    results = load_results(Path(args.results_path))

    for dataset in sorted(results):
        strategies = results[dataset]
        print(f"\n=== {dataset.upper()} (test de McNemar apparié, Exact Match) ===")
        pairs = [("fixed", "cast_orig"), ("cast_orig", "cast_scope"), ("fixed", "cast_scope")]
        for name_a, name_b in pairs:
            if name_a not in strategies or name_b not in strategies:
                continue
            mcnemar_test(name_a, name_b, strategies[name_a], strategies[name_b])


if __name__ == "__main__":
    main()
