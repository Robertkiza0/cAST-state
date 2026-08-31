"""Task loading for run_benchmark.py: RepoEval (already-cloned local repos)
and CrossCodeEval (task descriptions vendored locally, real repos cloned
on demand — same approach as repocoder-mine/colab_cceval_retrieval.ipynb's
ensure_repo_cloned, just as a reusable function instead of a notebook cell).
"""

import json
import random
import subprocess
from pathlib import Path
from typing import Any

from crosscodeeval_adapter import load_cceval_tasks, load_license_map, normalize_cceval_tasks

PROJECT_DIR = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    tasks = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def load_repoeval_tasks(
    tasks_path: Path = PROJECT_DIR / "datasets rapo" / "line_level_completion_1k_context_codegen.test.jsonl",
    repos_dir: Path = PROJECT_DIR / "data" / "repos_source",
    n_tasks: int = 50,
    tasks_per_repo: int = 10,
) -> list[dict[str, Any]]:
    """Sample up to n_tasks RepoEval tasks (at most tasks_per_repo per repo,
    only from repos already cloned in repos_dir), each augmented with
    `repo_dir` (Path) and `repo` (folder name) for the caller."""
    all_tasks = load_jsonl(tasks_path)
    seen_per_repo: dict[str, int] = {}
    selected = []

    for task in all_tasks:
        if len(selected) >= n_tasks:
            break
        metadata = task["metadata"]
        repo = metadata["task_id"].split("/")[0]
        if seen_per_repo.get(repo, 0) >= tasks_per_repo:
            continue
        repo_dir = repos_dir / repo
        if not repo_dir.exists():
            continue
        seen_per_repo[repo] = seen_per_repo.get(repo, 0) + 1
        selected.append({**task, "repo_dir": repo_dir, "repo": repo})

    return selected


def ensure_repo_cloned(owner_repo: str, cache_dir: Path, timeout_s: int = 120) -> Path | None:
    """Shallow-clone owner_repo (e.g. "turboderp/exllama") into cache_dir if
    not already cached there. Returns the local path, or None on failure
    (private/deleted repo, network error, timeout — CCEval task descriptions
    can reference repos that no longer resolve)."""
    safe_name = owner_repo.replace("/", "_")
    repo_dir = cache_dir / safe_name
    if repo_dir.exists():
        return repo_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{owner_repo}.git"
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, str(repo_dir)],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(f"Timeout en clonant {owner_repo}")
        return None

    if result.returncode != 0:
        print(f"Échec du clonage de {owner_repo}: {result.stderr.strip()[:200]}")
        return None
    return repo_dir


def load_cceval_tasks_sample(
    data_dir: Path = PROJECT_DIR / "crosscodeeval_data",
    n_tasks: int = 50,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample up to n_tasks normalized CCEval tasks (owner/repo resolved via
    the license map; real repo cloning happens lazily per-task in
    run_benchmark.py via ensure_repo_cloned, not here)."""
    license_map = load_license_map(data_dir / "LICENSES" / "project_license_map.txt")
    records = load_cceval_tasks(data_dir / "python" / "line_completion.jsonl")
    all_tasks = normalize_cceval_tasks(records, license_map)
    return random.Random(seed).sample(all_tasks, min(n_tasks, len(all_tasks)))
