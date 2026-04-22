# CNN Filters Initialization with DCT

## Objectif du projet

Ce projet étudie l’impact d’une initialisation structurée DCT-low pour les filtres 3×3 sur la convergence, la stabilité, la performance finale et l’évolution fréquentielle des filtres, en comparaison avec l’initialisation He classique. L’objectif est de comprendre si une représentation fréquentielle structurée influence la qualité de l’apprentissage sur CIFAR-10 avec un ResNet20.

## Résumé de la méthodologie

- Définition et organisation des expériences dans `configs/`.
- Construction du ResNet20 et implémentation de custom initializers dans `src/models/initializers.py`.
- Entraînement et évaluation avec des pipelines Python dans `src/pipelines/`.
- Analyse offline des filtres appris, des métriques et des composantes DCT dans `src/analysis/`.
- Validation par des tests `pytest` dans `tests/` et vérifications complémentaires dans `notebooks/`.

## Structure du dépôt

- `configs/` : initialisations entraînées, testées et analysées.
- `configs_hybrid/` : initialisations complexes/hybrides qui n’ont pas été testées.
- `src/` : fonctions et modules réutilisables du projet.
  - `src/data/` : chargement et préparation du dataset CIFAR-10.
  - `src/models/` : définition du ResNet20 et des initialisateurs personnalisés.
  - `src/training/` : boucle d’entraînement, évaluation et callbacks.
  - `src/analysis/` : analyse des filtres appris, DCT et métriques associées.
  - `src/pipelines/` : scripts de pipeline pour entraîner et analyser des runs.
  - `src/utils/` : utilitaires pour la configuration, l’I/O, le logging et la reproductibilité.
- `tests/` : plusieurs tests `pytest` pour assurer le bon fonctionnement des fonctions.
- `notebooks/` : notebooks complémentaires aux tests (sauf 03).
  - `01_sanity_checks.ipynb` : contrôles de base.
  - `02_filter_analysis.ipynb` : analyse détaillée des filtres.
  - `03_results_figures.ipynb` : génération des graphiques des runs.
- `runs/` : résultats et analyses de chaque exécution.
- `comparison/` : graphiques pour l’analyse inter-initialisation.
- `reports/` : figures et tables de synthèse.

## Installation des dépendances

Ce projet utilise Python 3.10.

1. Installer Anaconda / Miniconda.
2. Créer l’environnement :

```bash
conda env create -f environment.yml
conda activate cnn-dct-init
```

3. Installer les dépendances pip si nécessaire :

```bash
pip install -r requirements.txt
```

## Ordre recommandé d’exécution

1. Exécuter les tests unitaires :

```bash
pytest
```

2. Ouvrir `notebooks/01_sanity_checks.ipynb` pour valider les fonctions.
3. Ouvrir `notebooks/02_filter_analysis.ipynb` pour analyser les filtres.
4. Lancer `src/pipelines/train_and_analyze.py` ou `src/pipelines/train_and_analyze_batch.py` pour entraîner et analyser les runs (étape non effectuée dans les notebooks).
5. Ouvrir `notebooks/03_results_figures.ipynb` pour générer les graphiques des runs.

## Emplacement des résultats

- `runs/` : dossiers de chaque run avec checkpoints, métriques et fichiers de sortie.
- `comparison/` : graphiques d’analyse inter-initialisation.
- `reports/` : figures et tableaux de synthèse.

## Limites connues

- Les notebooks vérifient le comportement, mais ne couvrent pas l’entraînement complet.
- `src/pipelines/train_and_analyze.py` et `src/pipelines/train_and_analyze_batch.py` organisent l’entraînement et l’évaluation, mais ces étapes ne sont pas exécutées dans les notebooks.
- `configs_hybrid/` rassemble des initialisations complexes qui n’ont pas encore été testées.
- L’analyse reste centrée sur CIFAR-10 et ResNet20 ; d’autres jeux de données ou architectures ne sont pas traités.
- Les tests garantissent le fonctionnement des fonctions, pas forcément la robustesse de toutes les expérimentations.

## Notes importantes

- Le dossier `tests/` contient plusieurs tests `pytest` pour s’assurer du bon fonctionnement des fonctions.
- Le dossier `src/` contient l’ensemble des fonctions utilisées dans le projet et appelées depuis les notebooks.
- Le sous-dossier `src/pipelines/` regroupe les étapes d’entraînement et d’évaluation.
- Le dossier `runs/` contient les résultats et l’analyse de chaque run.
- Le dossier `notebooks/` est complémentaire à `tests/`, avec un ensemble de tests/analyses. Le notebook `03_results_figures.ipynb` est dédié à la création des graphiques des runs.
- Le dossier `comparison/` contient les graphiques destinés à l’analyse inter-initialisation.



