# cAST-state

Comparaison de stratégies de chunking pour la génération de code augmentée par
retrieval (RACG), extraite du projet `repocoder-mine`. Trois stratégies de
découpage sur le même retriever/scoring pondéré + attention AST :

- **`ast_chunker.py`** — fenêtres de lignes glissantes, enrichies des
  identifiants de portée (scope-aware), pas de respect strict des frontières AST.
- **`container_ast_chunker.py`** — découpage par conteneurs AST fait maison
  (fonction/classe entière si possible, sinon découpe par instruction de haut
  niveau interne avec en-tête + chevauchement conservés). Budget en lignes.
- **`cast_pipeline.py`** — adaptateur vers [`astchunk`](https://github.com/yilinjz/astchunk),
  l'implémentation officielle de l'algorithme **cAST** (Zhang et al., EMNLP
  2025 Findings — [papier](https://aclanthology.org/2025.findings-emnlp.430/)) :
  parcours récursif split-then-merge, taille mesurée en caractères
  non-blancs, fusion gloutonne des nœuds AST frères.

`weighted_ast_scorer.py` (Jaccard pondéré : IDF précalculé × attention
statique AST × poids par type de symbole) et `ast_distance.py` (attention
`exp(-λ·distance_ast)` par rapport au curseur) sont communs aux trois.

## Pourquoi

Le projet `repocoder-mine` a déjà confirmé (voir tests McNemar) que, sur les
mêmes chunks, un signal de scoring plus sophistiqué (identifiants AST, IDF+type,
attention par distance AST) ne bat pas significativement le Jaccard texte brut,
ni sur RepoCoder ni sur CrossCodeEval. L'axe encore ouvert est celui du
**chunking** lui-même : est-ce que découper par structure AST (au lieu de
fenêtres de lignes fixes) améliore le retrieval/la génération ? `cast_pipeline.py`
apporte l'implémentation *officielle* de cAST comme point de comparaison,
plutôt qu'une réimplémentation maison — pour pouvoir citer et comparer
directement au papier source.

## Installation

```bash
pip install -r requirements.txt
```

## Tests rapides (unitaires, sans données réelles)

```bash
python -m unittest test_container_ast_chunker test_ast_distance test_weighted_ast_scorer
```

## Comparaison sur données réelles (50 tâches RepoCoder, 5 dépôts)

Nécessite `data/repos_source/` (dépôts clonés) et
`datasets rapo/line_level_completion_1k_context_codegen.test.jsonl` — non
versionnés ici (voir `.gitignore`), à récupérer depuis `repocoder-mine` ou à
retélécharger depuis [microsoft/CodeT](https://github.com/microsoft/CodeT).

```bash
python test_weighted_ast_retrieval.py       # fenêtres glissantes
python test_container_weighted_retrieval.py # conteneurs AST maison
python test_cast_weighted_retrieval.py      # vrai cAST (astchunk)
```

## À faire

- Notebook Colab autonome (télécharge les dépôts RepoCoder lui-même) avec les
  3 conditions + test de significativité McNemar apparié, sur le modèle de
  `colab_container_ast_retrieval.ipynb` dans `repocoder-mine`.
- Décider si cAST est comparé seul contre les deux autres, ou si les 3 sont
  croisées avec les variantes de scoring (brut / IDF+type / +attention) déjà
  testées dans `repocoder-mine`.
