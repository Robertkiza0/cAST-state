"""Metrics for run_benchmark.py: Exact Match, Edit Similarity, Pass@1.

compute_em/compute_es are the same definitions used throughout this
project's other notebooks (repocoder-mine/colab_50_trials.ipynb etc.), kept
verbatim for comparability with earlier results on the same datasets.

Edit distance: prefers the compiled `editdistance` package (fast; installs
fine on the Colab/Linux boxes the rest of this project runs heavy jobs on)
but falls back to a pure-Python Levenshtein distance — same metric, just
slower — when no wheel is available (e.g. Python 3.13 on Windows here has
no prebuilt wheel and no local C compiler to build one from source).
"""

try:
    import editdistance

    def _edit_distance(a: str, b: str) -> int:
        return editdistance.eval(a, b)
except ImportError:
    def _edit_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        previous_row = list(range(len(b) + 1))
        for i, char_a in enumerate(a, start=1):
            current_row = [i] + [0] * len(b)
            for j, char_b in enumerate(b, start=1):
                current_row[j] = min(
                    current_row[j - 1] + 1,       # insertion
                    previous_row[j] + 1,          # deletion
                    previous_row[j - 1] + (char_a != char_b),  # substitution
                )
            previous_row = current_row
        return previous_row[-1]


def compute_em(target: str, prediction: str) -> int:
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
    return int(target_lines == prediction_lines and len(target_lines) > 0)


def compute_es(target: str, prediction: str) -> float:
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_str = "\n".join(target_lines)
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
    prediction_str = "\n".join(prediction_lines)
    if not target_str and not prediction_str:
        return 1.0
    return 1 - (_edit_distance(target_str, prediction_str) / max(len(target_str), len(prediction_str), 1))


def compute_pass_at_1(target: str, prediction: str) -> int:
    """Pass@1 for a single (greedy) sampled completion.

    LIMITATION, report alongside any number this produces: RepoEval's and
    CrossCodeEval's *line-level* completion splits (the only ones vendored in
    this project — data/repos_source + "datasets rapo/" for RepoEval,
    crosscodeeval_data/python for CCEval) carry no execution harness — no
    unit tests to run a completion against. Only RepoEval's separate
    function/API-level split is unit-test-backed, and it isn't part of this
    project's data. Without an execution harness, "pass" can only mean
    "the completion is correct", which for one greedy generation is exactly
    compute_em(). This function is therefore compute_em() under a research
    name for our datasets: it is NOT a substitute for the true
    execution-based Pass@1 the cAST paper reports on SWE-bench, and must not
    be presented as equivalent to it.
    """
    return compute_em(target, prediction)
