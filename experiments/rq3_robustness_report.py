"""RQ3 (Robustness to Structural Errors): empirical sweep over REAL corpus
files, truncated at random cut points to simulate a cursor mid-file — the
actual condition repository-level code completion operates under (the file
being completed is, by construction, not yet syntactically complete right
at the cursor).

For a sample of real files containing at least one class, each is cut at
several fractions of its length (25%/50%/75%/90%) and chunked with BOTH
cast_orig and cast_scope. Reports, per strategy:
  - crash rate (an uncaught exception escaping chunkify — NOT the same as
    a SyntaxError that chunk_file() already catches and handles by
    skipping the file; here we call chunkify() directly to see the raw
    exception, if any)
  - among the runs where a class ancestor's header was produced, whether
    cast_scope's self.* extraction crashed on the ERROR-containing subtree
    (a category of failure cast_orig cannot have, since it never walks the
    class body at all)

See test_rq3_robustness.py for the deterministic, synthetic-snippet unit
tests this complements.

Usage:
    python experiments/rq3_robustness_report.py --n-files 30 --seed 42
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

CUT_FRACTIONS = (0.25, 0.5, 0.75, 0.9)


def find_class_containing_files(repos_dir: Path, n_files: int, seed: int) -> list[Path]:
    candidates = []
    for repo_dir in sorted(p for p in repos_dir.iterdir() if p.is_dir()):
        for py_file in repo_dir.rglob("*.py"):
            try:
                text = py_file.read_text(encoding="utf-8")
            except OSError:
                continue
            if "class " in text and len(text) > 500:
                candidates.append(py_file)
    random.Random(seed).shuffle(candidates)
    return candidates[:n_files]


def try_chunk(builder, code: str) -> tuple[bool, str | None, list[dict] | None]:
    """Retourne (a_crashe, message_erreur, chunks). SyntaxError/ValueError
    sont les échecs de parsing "propres" que chunk_file() gère déjà en
    production (voir chunkers/__init__.py) — pas comptés comme un crash
    ici, seulement une VRAIE exception inattendue l'est."""
    try:
        chunks = builder.chunkify(code, chunk_expansion=True)
        return False, None, chunks
    except (SyntaxError, ValueError) as error:
        return False, f"parsing propre échoué (attendu, géré en prod): {error}", None
    except Exception as error:  # noqa: BLE001 - on veut voir TOUT le reste
        return True, f"{type(error).__name__}: {error}", None


def run_sweep(files: list[Path], max_chunk_size: int) -> dict[str, Any]:
    from astchunk import ASTChunkBuilder as OrigBuilder
    from astchunk_scope import ASTChunkBuilder as ScopeBuilder

    orig_builder = OrigBuilder(max_chunk_size=max_chunk_size, language="python", metadata_template="default")
    scope_builder = ScopeBuilder(max_chunk_size=max_chunk_size, language="python", metadata_template="default")

    n_runs = 0
    crashes = {"cast_orig": 0, "cast_scope": 0}
    clean_parses = {"cast_orig": 0, "cast_scope": 0}
    chunks_with_state = 0
    crash_details = []

    for file_path in files:
        try:
            full_code = file_path.read_text(encoding="utf-8")
        except OSError:
            continue

        for fraction in CUT_FRACTIONS:
            truncated = full_code[: int(len(full_code) * fraction)]
            if not truncated.strip():
                continue
            n_runs += 1

            orig_crashed, orig_msg, _ = try_chunk(orig_builder, truncated)
            scope_crashed, scope_msg, scope_chunks = try_chunk(scope_builder, truncated)

            if orig_crashed:
                crashes["cast_orig"] += 1
                crash_details.append({"file": str(file_path), "fraction": fraction, "strategy": "cast_orig", "error": orig_msg})
            else:
                clean_parses["cast_orig"] += 1

            if scope_crashed:
                crashes["cast_scope"] += 1
                crash_details.append({"file": str(file_path), "fraction": fraction, "strategy": "cast_scope", "error": scope_msg})
            else:
                clean_parses["cast_scope"] += 1
                if scope_chunks and any("(State:" in c["content"] for c in scope_chunks):
                    chunks_with_state += 1

    return {
        "n_files": len(files),
        "n_runs": n_runs,
        "crashes": crashes,
        "clean_parses": clean_parses,
        "crash_rate": {k: round(v / n_runs, 4) if n_runs else 0.0 for k, v in crashes.items()},
        "runs_with_extracted_state_despite_truncation": chunks_with_state,
        "crash_details": crash_details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repos-dir", default=str(PROJECT_DIR / "data" / "repos_source"))
    parser.add_argument("--n-files", type=int, default=30)
    parser.add_argument("--max-chunk-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default=str(PROJECT_DIR / "results"))
    args = parser.parse_args()

    files = find_class_containing_files(Path(args.repos_dir), args.n_files, args.seed)
    print(f"RQ3: {len(files)} fichiers réels échantillonnés (contenant au moins une classe)")
    print(f"Points de coupure simulant le curseur: {[f'{int(f*100)}%' for f in CUT_FRACTIONS]}\n")

    t0 = time.time()
    result = run_sweep(files, args.max_chunk_size)
    elapsed = time.time() - t0

    print(f"{result['n_runs']} exécutions (fichier x point de coupure)\n")
    print(f"{'Stratégie':<12} {'Crashs':>8} {'Parses propres':>16} {'Taux de crash':>15}")
    print("-" * 55)
    for strategy in ("cast_orig", "cast_scope"):
        print(
            f"{strategy:<12} {result['crashes'][strategy]:>8} "
            f"{result['clean_parses'][strategy]:>16} "
            f"{result['crash_rate'][strategy]:>14.2%}"
        )
    print(
        f"\ncast_scope a extrait un état de classe (self.*) malgré la troncature "
        f"dans {result['runs_with_extracted_state_despite_truncation']}/{result['n_runs']} exécutions."
    )
    if result["crash_details"]:
        print(f"\n{len(result['crash_details'])} détail(s) de crash — voir le fichier JSON.")
    print(f"\nTemps total: {elapsed:.1f}s")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"rq3_robustness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Résultats détaillés: {out_path}")


if __name__ == "__main__":
    main()
