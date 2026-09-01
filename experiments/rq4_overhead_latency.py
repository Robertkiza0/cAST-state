"""RQ4 (Overhead & Latency): What is the computational latency and token
overhead introduced by cAST-Scope during the chunking process and prompt
construction, compared to the baseline cAST (cast_orig)?

Two things are measured, per repo, cast_orig vs cast_scope:

1. Chunking latency: wall-clock time to chunk every .py file in the repo.
   cast_orig and cast_scope share the exact same windowing (see
   test_chunkers.py:test_cast_orig_and_cast_scope_have_identical_windowing)
   — the only code that differs between them is build_chunk_ancestors()
   (self.* extraction + decorator lookup). Any latency delta here is
   therefore attributable ONLY to that extra work, not to different chunk
   boundaries or a different underlying parse.

2. Token/char overhead: cast_scope's chunk_expansion header
   ("class Foo: (State: self.x, self.y)") is longer than cast_orig's plain
   one ("class Foo:"). This directly matters downstream — run_benchmark.py
   packs retrieved chunks into a fixed --max-context-chars budget, so a
   longer header per chunk means fewer whole chunks fit. Measured as extra
   characters in the chunk_expansion header, per chunk that has at least
   one class/function ancestor annotated (chunks with no ancestors, e.g.
   top-level module code, are identical between strategies and excluded).

Usage:
    python experiments/rq4_overhead_latency.py
    python experiments/rq4_overhead_latency.py --repos-dir data/repos_source --max-chunk-size 2000
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def measure_repo(repo_dir: Path, strategy: str, max_chunk_size: int) -> dict[str, Any]:
    if strategy == "cast_orig":
        from astchunk import ASTChunkBuilder
    elif strategy == "cast_scope":
        from astchunk_scope import ASTChunkBuilder
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    builder = ASTChunkBuilder(max_chunk_size=max_chunk_size, language="python", metadata_template="default")

    total_chunking_time_s = 0.0
    n_chunks = 0
    n_files = 0
    n_files_skipped = 0
    header_overhead_chars: list[int] = []

    for py_file in repo_dir.rglob("*.py"):
        try:
            code = py_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if not code.strip():
            continue

        # Chunker sans expansion : mesure la latence "pure" de chunking,
        # sans le coût de construction des chaînes d'en-tête (identique
        # entre les deux stratégies, ne doit pas polluer la comparaison).
        t0 = time.perf_counter()
        try:
            raw_windows = builder.chunkify(code, chunk_expansion=False)
        except (SyntaxError, ValueError):
            n_files_skipped += 1
            continue
        total_chunking_time_s += time.perf_counter() - t0
        n_files += 1
        n_chunks += len(raw_windows)

        # Rechunker avec expansion pour isoler le surcoût en caractères de
        # l'en-tête seul (même fenêtrage garanti, voir docstring du module).
        try:
            expanded_windows = builder.chunkify(code, chunk_expansion=True)
        except (SyntaxError, ValueError):
            continue
        for raw, expanded in zip(raw_windows, expanded_windows):
            overhead = len(expanded["content"]) - len(raw["content"])
            if overhead > 0:
                header_overhead_chars.append(overhead)

    return {
        "strategy": strategy,
        "n_files": n_files,
        "n_files_skipped": n_files_skipped,
        "n_chunks": n_chunks,
        "total_chunking_time_s": round(total_chunking_time_s, 4),
        "avg_time_per_chunk_ms": round((total_chunking_time_s / n_chunks) * 1000, 4) if n_chunks else 0.0,
        "avg_time_per_file_ms": round((total_chunking_time_s / n_files) * 1000, 4) if n_files else 0.0,
        "n_chunks_with_ancestor_header": len(header_overhead_chars),
        "avg_header_overhead_chars": round(sum(header_overhead_chars) / len(header_overhead_chars), 1) if header_overhead_chars else 0.0,
        "max_header_overhead_chars": max(header_overhead_chars, default=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repos-dir", default=str(PROJECT_DIR / "data" / "repos_source"))
    parser.add_argument("--max-chunk-size", type=int, default=2000)
    parser.add_argument("--results-dir", default=str(PROJECT_DIR / "results"))
    args = parser.parse_args()

    repos_dir = Path(args.repos_dir)
    repos = sorted(p for p in repos_dir.iterdir() if p.is_dir())
    if not repos:
        raise SystemExit(f"Aucun dépôt trouvé dans {repos_dir}")

    print(f"RQ4: {len(repos)} dépôts, max_chunk_size={args.max_chunk_size}\n")

    all_results: dict[str, dict[str, Any]] = {}
    for repo_dir in repos:
        print(f"--- {repo_dir.name} ---")
        repo_results = {}
        for strategy in ("cast_orig", "cast_scope"):
            t0 = time.perf_counter()
            stats = measure_repo(repo_dir, strategy, args.max_chunk_size)
            wall = time.perf_counter() - t0
            repo_results[strategy] = stats
            print(
                f"  {strategy:<12} {stats['n_chunks']:>6} chunks | "
                f"{stats['avg_time_per_chunk_ms']:>6.3f} ms/chunk | "
                f"overhead en-tête moyen: {stats['avg_header_overhead_chars']:>5.1f} car. | "
                f"({wall:.1f}s au total)"
            )
        all_results[repo_dir.name] = repo_results

    # Agrégats globaux (tous dépôts confondus).
    print("\n" + "=" * 70)
    print("AGRÉGAT (tous dépôts)")
    print("=" * 70)
    for strategy in ("cast_orig", "cast_scope"):
        total_chunks = sum(r[strategy]["n_chunks"] for r in all_results.values())
        total_time = sum(r[strategy]["total_chunking_time_s"] for r in all_results.values())
        overhead_values = [r[strategy]["avg_header_overhead_chars"] for r in all_results.values() if r[strategy]["n_chunks_with_ancestor_header"] > 0]
        avg_overhead = sum(overhead_values) / len(overhead_values) if overhead_values else 0.0
        print(
            f"{strategy:<12} {total_chunks:>7} chunks | {total_time:>7.2f}s total | "
            f"{(total_time/total_chunks*1000) if total_chunks else 0:>6.3f} ms/chunk moyen | "
            f"overhead en-tête moyen: {avg_overhead:>5.1f} car./chunk"
        )
    print("=" * 70)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"rq4_overhead_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRésultats détaillés: {out_path}")


if __name__ == "__main__":
    main()
