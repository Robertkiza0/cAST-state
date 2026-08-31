"""Compare 3 chunking baselines (fixed-size, cAST officiel, cAST-Scope) sur
RepoEval et/ou CrossCodeEval, avec le MÊME retriever (BM25) et le MÊME
générateur pour les 3, afin d'isoler l'effet du chunking seul.

Baselines :
    fixed      - Fenêtre fixe de texte (max_chunk_size caractères), aucune
                 conscience de l'AST. La baseline de base du papier cAST.
    cast_orig  - cAST officiel (astchunk, pip install astchunk, non modifié) :
                 build_chunk_ancestors naïf (split('\\n')[0]).
    cast_scope - cAST-Scope (astchunk_scope/, ce projet) : ancêtres enrichis
                 de l'état self.* des classes et des décorateurs des fonctions.

Exemple (test rapide, sans GPU, générateur factice pour valider tout le pipeline) :
    python run_benchmark.py --dataset repoeval --n-tasks 10 --generator stub

Exemple (run réel, sur une machine GPU) :
    python run_benchmark.py --dataset both --n-tasks 300 --generator hf \\
        --model-name bigcode/starcoder2-7b --device cuda

LIMITATION (voir metrics.py:compute_pass_at_1) : les deux datasets vendorisés
ici sont des variantes de complétion "line-level" sans harnais d'exécution ;
Pass@1 y est donc numériquement identique à Exact Match, pas une vraie mesure
d'exécution comme dans le papier cAST sur SWE-bench. C'est signalé dans le
tableau final, pas caché.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from chunkers import STRATEGIES, chunk_file
from crosscodeeval_adapter import filter_safe_chunks_cceval, postprocess_completion
from datasets_io import ensure_repo_cloned, load_cceval_tasks_sample, load_repoeval_tasks
from generation import get_generator
from metrics import compute_em, compute_es, compute_pass_at_1
from retrieval import BM25Retriever

PROJECT_DIR = Path(__file__).resolve().parent

# Coupe l'unfinished_code AVANT de construire le prompt complet (jamais après,
# jamais laissé au tokenizer) : c'est le correctif d'un bug déjà rencontré
# dans ce projet (repocoder-mine/colab_50_trials.ipynb) où une troncature
# côté tokenizer (par défaut à droite) coupait silencieusement la fin du
# prompt — justement la partie la plus proche du point de complétion.
DEFAULT_MAX_UNFINISHED_CHARS = 1500


def trim_code(code: str, max_chars: int = DEFAULT_MAX_UNFINISHED_CHARS) -> str:
    return code[-max_chars:] if len(code) > max_chars else code


def build_prompt(unfinished_code: str, retrieved_chunks: list[dict[str, Any]], max_context_chars: int = 3000) -> str:
    context_parts = []
    total = 0
    for chunk in retrieved_chunks:
        content = chunk["content"]
        if total + len(content) > max_context_chars:
            remaining = max_context_chars - total
            if remaining > 0:
                context_parts.append(content[:remaining])
            break
        context_parts.append(content)
        total += len(content)

    context = "\n\n".join(context_parts)
    trimmed = trim_code(unfinished_code)
    return f"{context}\n\n{trimmed}" if context else trimmed


def filter_safe_chunks_repoeval(
    chunks: list[dict[str, Any]], repo_dir: Path, fpath_tuple: list[str], context_start_lineno: int
) -> list[dict[str, Any]]:
    task_file = os.path.normpath(str(repo_dir.joinpath(*fpath_tuple[1:])))
    safe = []
    for chunk in chunks:
        if os.path.normpath(chunk["file_path"]) == task_file and chunk["end_line"] - 1 > context_start_lineno:
            continue
        safe.append(chunk)
    return safe


class RepoChunkCache:
    """Chunke chaque dépôt (par stratégie) une seule fois, réutilisé par
    toutes les tâches de ce dépôt — chunker un dépôt entier par tâche serait
    un gaspillage énorme (des dizaines de tâches partagent le même dépôt)."""

    def __init__(self, max_chunk_size: int):
        self.max_chunk_size = max_chunk_size
        self._cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def get(self, repo_dir: Path, strategy: str) -> list[dict[str, Any]]:
        key = (str(repo_dir), strategy)
        if key not in self._cache:
            print(f"[chunking] Début {strategy} sur {repo_dir}...")
            # "fixed" never hangs (pure Python, no tree-sitter) — only the AST
            # strategies go through tree-sitter, which is where a hang was
            # traced to. Print BEFORE each file (not after, and not just every
            # N files) for those specifically: if one file hangs, the last
            # line printed is unambiguously that exact file, not a guess
            # bracketed by whichever heartbeat interval happened to be set.
            # TEMPORARY while root-causing a real stall seen on Colab but not
            # reproduced locally (same repo) — dial back once the culprit
            # file/pattern is identified and handled properly.
            trace_every_file = strategy != "fixed"
            chunks = []
            n_files = 0
            for root, _, files in os.walk(repo_dir):
                for filename in files:
                    if filename.endswith(".py"):
                        file_path = os.path.join(root, filename)
                        try:
                            code = open(file_path, "r", encoding="utf-8").read()
                        except OSError:
                            continue
                        n_files += 1
                        if trace_every_file:
                            print(f"[chunking]   ({strategy}) fichier {n_files}: {file_path} ({len(code)} car.)")
                        elif n_files % 200 == 0:
                            print(f"[chunking]   ({strategy}) {n_files} fichiers traités, en cours: {file_path}")
                        t0 = time.time()
                        chunks.extend(chunk_file(file_path, code, strategy, self.max_chunk_size))
                        elapsed = time.time() - t0
                        if elapsed > 2.0:
                            # Repère un fichier pathologique pour tree-sitter (ex. ligne générée
                            # anormalement longue) plutôt que de laisser le run entier sembler figé.
                            longest_line = max((len(line) for line in code.splitlines()), default=0)
                            print(
                                f"[chunking] LENT: {file_path} ({strategy}) en {elapsed:.1f}s "
                                f"— {len(code)} caractères, ligne la plus longue: {longest_line}"
                            )
            self._cache[key] = chunks
        return self._cache[key]


def new_result_bucket() -> dict[str, list]:
    return {"em": [], "es": [], "pass1": []}


def score(target: str, prediction: str, bucket: dict[str, list]) -> None:
    bucket["em"].append(compute_em(target, prediction))
    bucket["es"].append(compute_es(target, prediction))
    bucket["pass1"].append(compute_pass_at_1(target, prediction))


def run_repoeval(args, chunk_cache: RepoChunkCache, generator, results: dict) -> None:
    tasks = load_repoeval_tasks(n_tasks=args.n_tasks, tasks_per_repo=args.tasks_per_repo)
    print(f"[RepoEval] {len(tasks)} tâches chargées ({len(set(t['repo'] for t in tasks))} dépôts)")

    for i, task in enumerate(tasks, 1):
        metadata = task["metadata"]
        repo_dir = task["repo_dir"]
        verbose = i == 1  # instrumentation détaillée sur la toute première tâche seulement

        if verbose:
            print(f"[RepoEval] Tâche 1: dépôt={task['repo']}, début du traitement...")

        for strategy in STRATEGIES:
            t0 = time.time()
            chunks = chunk_cache.get(repo_dir, strategy)
            if verbose:
                print(f"[RepoEval]   [{strategy}] chunking dépôt: {len(chunks)} chunks en {time.time()-t0:.1f}s")

            t0 = time.time()
            safe_chunks = filter_safe_chunks_repoeval(
                chunks, repo_dir, metadata["fpath_tuple"], metadata["context_start_lineno"]
            )
            retriever = BM25Retriever(safe_chunks)
            retrieved = retriever.retrieve(task["prompt"], k=args.k)
            if verbose:
                print(f"[RepoEval]   [{strategy}] retrieval: {len(retrieved)} chunks en {time.time()-t0:.1f}s")

            t0 = time.time()
            prompt = build_prompt(task["prompt"], retrieved, max_context_chars=args.max_context_chars)
            prediction = generator.generate(prompt)
            if verbose:
                print(f"[RepoEval]   [{strategy}] generate(): {time.time()-t0:.1f}s, sortie={prediction[:60]!r}")

            score(metadata["ground_truth"], prediction, results["repoeval"][strategy])

        if verbose:
            print("[RepoEval] Tâche 1 terminée avec succès.")

        if i % 5 == 0 or i == len(tasks):
            print(f"[RepoEval] {i}/{len(tasks)} tâches traitées")


def run_cceval(args, chunk_cache: RepoChunkCache, generator, results: dict) -> None:
    tasks = load_cceval_tasks_sample(n_tasks=args.n_tasks, seed=args.seed)
    print(f"[CrossCodeEval] {len(tasks)} tâches normalisées échantillonnées")
    cache_dir = PROJECT_DIR / args.cceval_repo_cache

    processed = 0
    for i, task in enumerate(tasks, 1):
        metadata = task["metadata"]
        repo_dir = ensure_repo_cloned(metadata["owner_repo"], cache_dir)
        if repo_dir is None:
            continue

        for strategy in STRATEGIES:
            chunks = chunk_cache.get(repo_dir, strategy)
            safe_chunks = filter_safe_chunks_cceval(chunks, repo_dir, metadata["fpath_tuple"])
            retriever = BM25Retriever(safe_chunks)
            retrieved = retriever.retrieve(task["prompt"], k=args.k)
            prompt = build_prompt(task["prompt"], retrieved, max_context_chars=args.max_context_chars)
            raw_prediction = generator.generate(prompt)
            prediction = postprocess_completion(task["prompt"], raw_prediction)
            score(metadata["ground_truth"], prediction, results["cceval"][strategy])

        processed += 1
        if processed % 5 == 0:
            print(f"[CrossCodeEval] {processed}/{len(tasks)} tâches traitées (clonées avec succès)")

    print(f"[CrossCodeEval] {processed}/{len(tasks)} tâches traitées au total")


def print_comparison_table(results: dict, datasets_run: list[str]) -> None:
    def mean(values: list) -> float:
        return sum(values) / len(values) if values else float("nan")

    print("\n" + "=" * 78)
    print("RÉSULTATS")
    print("=" * 78)
    for dataset in datasets_run:
        print(f"\n{dataset.upper()}  (n={len(results[dataset][STRATEGIES[0]]['em'])} tâches scorées)")
        print(f"{'Baseline':<12} {'EM':>8} {'ES':>8} {'Pass@1':>8}")
        print("-" * 40)
        for strategy in STRATEGIES:
            bucket = results[dataset][strategy]
            print(
                f"{strategy:<12} {mean(bucket['em']):>8.3f} {mean(bucket['es']):>8.3f} {mean(bucket['pass1']):>8.3f}"
            )
    print(
        "\nNote: Pass@1 == EM ici (pas de harnais d'exécution sur ces variantes "
        "line-level — voir metrics.py:compute_pass_at_1)."
    )
    print("=" * 78)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["repoeval", "cceval", "both"], default="both")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--tasks-per-repo", type=int, default=10, help="RepoEval uniquement")
    parser.add_argument("--max-chunk-size", type=int, default=2000, help="Caractères non-blancs (identique aux 3 baselines)")
    parser.add_argument("--k", type=int, default=5, help="Nombre de chunks retrouvés par tâche")
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--generator", choices=["stub", "hf"], default="stub")
    parser.add_argument("--model-name", default=None, help="Requis si --generator hf, ex: bigcode/starcoder2-7b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cceval-repo-cache", default="cceval_repos")
    return parser.parse_args()


def main():
    # Python fully buffers stdout (instead of flushing per line) whenever
    # it isn't attached to a terminal — e.g. piped through `tee`, redirected
    # to a file, or captured by some notebook frontends. Force line
    # buffering so progress prints show up immediately regardless, and
    # — more importantly — so a crash (OOM-kill, CUDA abort) doesn't
    # silently swallow whatever was still sitting in the stdout buffer.
    # Guarded: under IPython's %run, sys.stdout is an OutStream wrapper
    # without .reconfigure() (it's already line-flushed by IPython itself).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    args = parse_args()
    random.seed(args.seed)

    if args.generator == "stub":
        print(
            "ATTENTION: --generator stub produit des résultats FACTICES "
            "(validation du pipeline uniquement, pas un vrai run pour le papier)."
        )

    generator = get_generator(args.generator, model_name=args.model_name, device=args.device)
    chunk_cache = RepoChunkCache(max_chunk_size=args.max_chunk_size)

    datasets_run = ["repoeval", "cceval"] if args.dataset == "both" else [args.dataset]
    results = {dataset: {strategy: new_result_bucket() for strategy in STRATEGIES} for dataset in datasets_run}

    start = time.time()
    try:
        if "repoeval" in datasets_run:
            run_repoeval(args, chunk_cache, generator, results)
        if "cceval" in datasets_run:
            run_cceval(args, chunk_cache, generator, results)
    finally:
        # Under `!python`, the OS reclaims all of this when the subprocess
        # exits. Under IPython's `%run` (same long-lived kernel process —
        # needed on Colab, see the notebook note above `%run`), it doesn't:
        # an HFGenerator's model would stay resident in VRAM/RAM after this
        # call returns, and the next %run'd model load would stack on top of
        # it instead of starting from a clean slate. `finally` so this still
        # runs even if the benchmark loop raises.
        if hasattr(generator, "model"):
            del generator.model
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

    print_comparison_table(results, datasets_run)
    print(f"\nTemps total: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
