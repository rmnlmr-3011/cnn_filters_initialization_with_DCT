# src/

Le dossier `src/` contient l’implémentation principale du projet. Il regroupe les modules suivants :

- `data/` : chargement et préparation du dataset CIFAR-10.
- `models/` : définition du modèle ResNet20 et des initialisateurs personnalisés.
- `training/` : boucle d’entraînement, évaluation, callbacks et gestion des checkpoints.
- `analysis/` : analyse des filtres, métriques DCT et traitement des checkpoints.
- `analysis_inter_model/` : outils d’analyse inter-modèle pour générer des graphiques comparatifs.
- `pipelines/` : scripts de pipeline pour exécuter l’entraînement et l’analyse.
- `utils/` : utilitaires de configuration, d’I/O, de logging et de gestion de la graine.

## Fichiers racine

- `__init__.py` : rend le package `src` importable comme module Python.

## data/

- `__init__.py` : chemin d’import du package de données.
- `loaders.py` : fonctions pour charger et préparer le dataset CIFAR-10, incluant le découpage entre train/validation.

## models/

- `__init__.py` : package modèles.
- `resnet20.py` : définit l’architecture ResNet20 utilisée pour les expériences.
- `initializers.py` : implémente les initialisations personnalisées de filtres, dont :
  - `he`
  - `sigma_pure`
  - `sigma`
  - `grad`
  - `grad_vertical_pure`
  - `grad_horizontal_pure`
  - `dctlow`
  - `dctlow_noise`

## training/

- `__init__.py` : package de formation.
- `callbacks.py` : callbacks Keras/TensorFlow personnalisés pour l’entraînement.
- `checkpoint_callbacks.py` : callbacks dédiés au sauvegarde de checkpoints spécifiques à chaque époque.
- `evaluate.py` : fonctions d’évaluation du modèle et calcul des métriques de performance.
- `train_core.py` : fonction centrale d’entraînement, orchestration des datasets, du modèle, du logging et des sauvegardes.
- `train.py` : script d’entrée pour démarrer un entraînement depuis la ligne de commande.

## analysis/

- `__init__.py` : package d’analyse.
- `dct.py` : transformée DCT et transformée inverse utilisées pour manipuler les filtres et leurs coefficients.
- `dct_bases.py` : définitions de masques et de structures basées sur les composantes DCT basses fréquences.
- `filter_metrics.py` : calcul des métriques de filtres et des mesures DCT comme l’énergie totale, le ratio basse fréquence et `beta_sq`.
- `filter_analysis.py` : analyse des poids appris d’un modèle, application des métriques et génération de résumés de kernels.
- `model_filters.py` : fonctions utilitaires pour l’extraction des couches convolutives 3×3 et l’itération sur les kernels.
- `checkpoint_analysis.py` : analyse des checkpoints d’un run et calcul de métriques sur les modèles sauvegardés.
- `checkpoint_series.py` : construction de séries temporelles et graphiques à partir de checkpoints.
- `run_filter_analysis.py` : script de commande pour lancer une analyse de filtres sur un run donné.
- `run_checkpoint_analysis.py` : script de commande pour analyser tous les checkpoints d’un run.
- `run_checkpoint_series.py` : script de commande pour construire et tracer des séries de checkpoints.

## analysis_inter_model/

- `__init__.py` : package d’analyse inter-modèle.
- `build_dataset.py` : création de jeux de données d’analyse comparant plusieurs runs ou configurations.
- `plot_barplots.py` : génération de barplots comparatifs entre initialisations.
- `plot_correlations.py` : calcul et tracé des corrélations entre métriques ou représentations de filtres.
- `plot_curves.py` : génération de courbes comparatives des performances et métriques.

## pipelines/

- `__init__.py` : package de pipelines.
- `train_and_analyze.py` : pipeline principale pour entraîner un modèle et lancer son analyse de filtres.
- `train_and_analyze_batch.py` : pipeline de traitement par lots pour exécuter plusieurs runs et analyses successifs.

## utils/

- `__init__.py` : package utilitaire.
- `config.py` : chargement et parsing des fichiers YAML de configuration.
- `io.py` : fonctions d’entrée/sortie, sauvegarde de résultats, génération de répertoires de run et export JSON/CSV.
- `logging.py` : configuration et gestion du logging du projet.
- `seed.py` : gestion de la graine aléatoire pour assurer la reproductibilité des runs.

## Usage générale

Le dossier `src/` contient l’ensemble des fonctions utilisées dans le projet et sert de cœur à l’entraînement, l’analyse et la génération de graphiques. Les notebooks et les tests appellent ces modules pour valider et visualiser les résultats.