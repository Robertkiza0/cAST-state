"""Generates a real, illustrative example of what a prompt actually looks
like under each of the 3 chunking baselines, for the same RepoEval task —
paper figure material (mirroring the cAST paper's own Figure 1: same
underlying code, different chunking/annotation, side by side).

Searches real tasks for one where the top-retrieved cast_scope chunk
carries a non-empty "(State: ...)" annotation (otherwise fixed/cast_orig/
cast_scope prompts look identical except for the presence/absence of a
plain "Scope:" line — illustrative, but not the interesting case). Falls
back to the first task with ANY class/function ancestor if no
State-carrying example is found within --n-tasks.

Usage:
    python experiments/example_prompts.py
    python experiments/example_prompts.py --n-tasks 200 --k 1 --output-file results/example.txt
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from chunkers import STRATEGIES
from datasets_io import load_repoeval_tasks
from retrieval import BM25Retriever
from run_benchmark import RepoChunkCache, build_prompt, filter_safe_chunks_repoeval


def find_illustrative_task(tasks: list, chunk_cache: RepoChunkCache, k: int) -> tuple[dict, list[dict]] | None:
    """Renvoie (task, chunks_retrouves_cast_scope) pour la première tâche où
    au moins un chunk retrouvé porte une annotation (State: ...). None si
    aucune ne convient (retombe sur le premier chunk avec un ancêtre tout
    court, pas forcément avec un état)."""
    fallback = None
    for task in tasks:
        metadata = task["metadata"]
        chunks = chunk_cache.get(task["repo_dir"], "cast_scope")
        safe_chunks = filter_safe_chunks_repoeval(
            chunks, task["repo_dir"], metadata["fpath_tuple"], metadata["context_start_lineno"]
        )
        retriever = BM25Retriever(safe_chunks)
        retrieved = retriever.retrieve(task["prompt"], k=k)
        if any("State:" in c.get("header", "") for c in retrieved):
            return task, retrieved
        if fallback is None and any(c.get("header") for c in retrieved):
            fallback = (task, retrieved)
    return fallback


def render_comparison(task: dict, k: int, max_context_chars: int, chunk_cache: RepoChunkCache) -> str:
    metadata = task["metadata"]
    lines = [
        f"=== TÂCHE: {metadata['task_id']} ===",
        f"ground_truth: {metadata['ground_truth']!r}",
        "",
    ]
    for strategy in STRATEGIES:
        chunks = chunk_cache.get(task["repo_dir"], strategy)
        safe_chunks = filter_safe_chunks_repoeval(
            chunks, task["repo_dir"], metadata["fpath_tuple"], metadata["context_start_lineno"]
        )
        retriever = BM25Retriever(safe_chunks)
        retrieved = retriever.retrieve(task["prompt"], k=k)
        prompt = build_prompt(task["prompt"], retrieved, max_context_chars=max_context_chars)
        lines.append(f"========== {strategy.upper()} ({len(prompt)} car.) ==========")
        lines.append(prompt)
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tasks", type=int, default=80, help="Tâches parcourues pour trouver un exemple illustratif")
    parser.add_argument("--tasks-per-repo", type=int, default=10)
    parser.add_argument("--k", type=int, default=1, help="Chunks retrouvés par prompt (1 = l'exemple le plus lisible)")
    parser.add_argument("--max-chunk-size", type=int, default=2000)
    parser.add_argument("--max-context-chars", type=int, default=800)
    parser.add_argument("--output-file", default=None, help="Si fourni, sauvegarde aussi dans ce fichier (sinon juste affiché)")
    args = parser.parse_args()

    tasks = load_repoeval_tasks(n_tasks=args.n_tasks, tasks_per_repo=args.tasks_per_repo)
    chunk_cache = RepoChunkCache(max_chunk_size=args.max_chunk_size)

    found = find_illustrative_task(tasks, chunk_cache, args.k)
    if found is None:
        print(f"Aucune tâche avec ancêtre trouvée parmi les {len(tasks)} tâches essayées — augmentez --n-tasks.")
        return
    task, _ = found

    output = render_comparison(task, args.k, args.max_context_chars, chunk_cache)
    print(output)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"\nSauvegardé: {out_path}")


if __name__ == "__main__":
    main()
