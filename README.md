# cAST-Scope

Résout la Limitation #1 officielle du papier cAST ("Contextual Awareness") :
enrichir la représentation des ancêtres de chaque chunk avec la portée d'état
de la classe englobante (attributs `self.*`) et les décorateurs de la
fonction englobante, sans dégrader la vitesse de chunking.

## Questions de recherche (RQ1–RQ4)

| RQ | Question | Où |
|---|---|---|
| **RQ1** | L'enrichissement des ancêtres AST (`self.*`, décorateurs) améliore-t-il la complétude structurelle par rapport aux métadonnées cAST standard ? | `astchunk_scope/`, `test_astchunk_scope.py` |
| **RQ2** | Comment cAST-Scope se compare-t-il au chunking line-based et à cAST baseline en EM/ES/Pass@1 ? | `run_benchmark.py`, `colab_*.ipynb` |
| **RQ3** | Le mécanisme d'expansion de portée reste-t-il stable face à du code incomplet/erroné près du curseur ? | `test_rq3_robustness.py`, `experiments/rq3_robustness_report.py` |
| **RQ4** | Quel est le surcoût en latence et en tokens introduit par cAST-Scope ? | `experiments/rq4_overhead_latency.py` |

## Structure

- **`astchunk_scope/`** — fork modifiable de [`astchunk`](https://github.com/yilinjz/astchunk)
  (Zhang et al., EMNLP 2025 Findings — [papier](https://aclanthology.org/2025.findings-emnlp.430/)).
  Seul changement de comportement : `ASTChunk.build_chunk_ancestors()` dans
  `astchunk_scope/astchunk.py` (voir docstring de `astchunk_scope/__init__.py`
  pour le détail, y compris le cache par appel à `chunkify()` qui garde le
  coût amorti sous 1 ms/chunk — voir `test_astchunk_scope.py`). Nommé
  `astchunk_scope` (pas `astchunk`) exprès : le paquet pip `astchunk` original
  reste intact et importable en parallèle, nécessaire pour la baseline
  `cast_orig` non modifiée.
- **`chunkers/`** — point d'entrée unifié `chunk_file(path, code, strategy, max_chunk_size)`
  pour les 3 baselines (`fixed`, `cast_orig`, `cast_scope`), même format de
  sortie quelle que soit la stratégie.
- **`retrieval/`** — `BM25Retriever`, partagé par les 3 baselines (seul le
  chunking varie entre les conditions).
- **`generation/`** — générateur enfichable : `HFGenerator` (vrai modèle
  HuggingFace, StarCoder2-7B/CodeLlama-7B — nécessite torch+GPU, à lancer sur
  Colab) ou `StubGenerator` (factice, déterministe, pour valider tout le
  pipeline sans GPU).
- **`metrics.py`** — EM, ES (Levenshtein, avec repli pur Python si le paquet
  compilé `editdistance` n'est pas installable), Pass@1 (voir limitation
  ci-dessous).
- **`datasets_io.py`** / **`crosscodeeval_adapter.py`** — chargement des
  tâches RepoEval (dépôts déjà clonés localement) et CrossCodeEval (tâches
  vendorisées localement, vrais dépôts clonés à la demande).
- **`run_benchmark.py`** — orchestre tout : compare les 3 baselines sur
  RepoEval et/ou CrossCodeEval avec le même retriever et le même générateur (RQ2).
- **`experiments/rq3_robustness_report.py`** — RQ3 : troncature de vrais
  fichiers à plusieurs points (25/50/75/90%, simule un curseur en cours de
  frappe), mesure le taux de crash cast_orig vs cast_scope et la capacité
  à extraire l'état de classe malgré la troncature. Complété par les tests
  synthétiques déterministes de `test_rq3_robustness.py`.
- **`experiments/rq4_overhead_latency.py`** — RQ4 : latence de chunking et
  surcoût en caractères de l'en-tête `chunk_expansion`, cast_orig vs
  cast_scope, sur tout le corpus local.

## Installation

```bash
pip install -r requirements.txt
```

## Tests (rapides, aucune donnée réelle nécessaire)

```bash
python -m unittest discover -p "test_*.py"
```

48 tests couvrent : l'annotation scope-aware (état de classe, décorateurs,
non-régression, performance amortie < 1 ms/chunk), le chunker fixe, le
retriever BM25, les métriques, la cohérence de l'interface unifiée entre
les 3 stratégies (`cast_orig`/`cast_scope` doivent avoir le MÊME fenêtrage,
seul le texte d'en-tête diffère), et la robustesse face au code
incomplet/erroné (RQ3).

## Lancer les expériences RQ3/RQ4 (locales, pas besoin de GPU)

```bash
python experiments/rq3_robustness_report.py --n-files 30
python experiments/rq4_overhead_latency.py
```

## Lancer le benchmark

Test rapide, sans GPU, générateur factice (valide tout le pipeline —
chunking, retrieval, prompt, scoring, tableau comparatif — sans télécharger
de modèle) :

```bash
python run_benchmark.py --dataset repoeval --n-tasks 10 --generator stub
python run_benchmark.py --dataset cceval --n-tasks 10 --generator stub
```

Run réel (sur une machine GPU) :

```bash
python run_benchmark.py --dataset both --n-tasks 300 --generator hf \
    --model-name bigcode/starcoder2-7b --device cuda
```

Sur Colab, deux notebooks prêts à l'emploi (clonent ce dépôt eux-mêmes) :
- `colab_quick_test.ipynb` — T4, 50 tâches RepoEval, un seul modèle (test rapide, économe en unités de calcul)
- `colab_benchmark.ipynb` — A100, 300 tâches, RepoEval+CrossCodeEval, StarCoder2+CodeLlama (run complet)

Toutes les options : `python run_benchmark.py --help`.

### Résultats

Sauvegardés au fur et à mesure dans `results/` (pas seulement à la fin) :
un `.jsonl` avec le détail par tâche/stratégie (écrit ligne par ligne, flush
immédiat — survit à un plantage en cours de route) et un `_summary.json`
avec les moyennes agrégées, écrit une fois le run terminé. Le chemin exact
est affiché au tout début de chaque run.

## Limitation connue : Pass@1

RepoEval et CrossCodeEval ne sont vendorisés ici que dans leur variante
*line-level completion*, sans harnais d'exécution (pas de tests unitaires à
exécuter contre la complétion générée). `compute_pass_at_1()` est donc, pour
ces deux datasets, numériquement identique à Exact Match — ce n'est PAS
l'équivalent du vrai Pass@1 basé sur exécution que le papier cAST rapporte
sur SWE-bench. `run_benchmark.py` l'affiche quand même (demandé dans la
spec), avec cette note en clair dans le tableau final.

## Données locales (non versionnées, voir `.gitignore`)

- `data/repos_source/` — 8 dépôts RepoEval déjà clonés.
- `datasets rapo/` — tâches RepoEval (jsonl).
- `crosscodeeval_data/python/` — tâches CrossCodeEval (jsonl) + carte des
  licences ; les vrais dépôts sont clonés à la demande dans `cceval_repos/`
  (créé automatiquement par `run_benchmark.py --dataset cceval`).
- `astchunk_reference/`, `cceval/` — clones de référence en lecture seule
  (code tiers, pas notre travail).
