# notebooks/

Ce dossier contient des notebooks Jupyter pour tester, vérifier et visualiser les résultats du projet.

## Objectif

Les notebooks offrent un environnement interactif pour :
- Valider le chargement des données et la construction du modèle.
- Analyser en détail les filtres et leurs propriétés DCT.
- Générer des graphiques comparatifs entre les différentes initialisations.

## Fichiers

### 01_sanity_checks.ipynb

**Objectif** : Vérifie les éléments fondamentaux du projet.

**Contenu** :
- Importation des bibliothèques TensorFlow et modules du projet.
- Chargement du dataset CIFAR-10 et découpage train/validation.
- Vérification des formes des données (x_train, y_train, etc.).
- Construction d'un modèle ResNet20 avec architecture standard.
- Compilation du modèle avec un optimiseur SGD.

**À exécuter** : En premier, pour confirmer que l'environnement est bien configuré.

### 02_filter_analysis.ipynb

**Objectif** : Analyse détaillée des filtres et des métriques DCT.

**Contenu** :
- Importation des fonctions d'analyse DCT, métriques et extraction de filtres.
- Chargement d'un modèle pré-entraîné depuis le dossier `runs/`.
- Extraction des couches convolutives 3×3 du modèle.
- Tests de validation :
  - Reconstruction DCT/IDCT sur un kernel 3×3.
  - Vérification de la conservation de l'énergie (Parseval).
  - Vérification de la symétrie et antisymétrie des kernels.
  - Calcul du ratio d'énergie basse fréquence.
  - Visualisation de la décomposition fréquentielle (DC, σ-gradient, gradients verticaux/horizontaux).
  - Génération de résumés de métriques par kernel et couche.

**À exécuter** : Après le notebook 1, pour valider que l'analyse fonctionne correctement.

### 03_results_figures.ipynb

**Objectif** : Génération de graphiques finaux et synthèse des résultats.

**Contenu** :
- Construction de jeux de données à partir de tous les runs du dossier `runs/`.
- Génération de **courbes comparatives** : affichage des métriques moyennes par initialisation au fil des runs.
- Génération de **barplots** : comparaison des performances finales par initialisation.
- Génération de **diagrammes de corrélation** : analyse des relations entre métriques et performance.
- Génération de **graphiques d'énergie DCT** : heatmaps comparant l'énergie de chaque composante DCT par initialisation.

Les graphiques sont sauvegardés dans `comparison/curves/`, `comparison/barplots/`, `comparison/correlation/`, et `comparison/dct_component_energy_barplots/`.

**À exécuter** : Après avoir généré plusieurs runs et analyses dans `runs/`.

## Dépendances

- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Modules du projet (`src/`)

## Remarques

- Ces notebooks ne lancent PAS d'entraînement complet, seulement des analyses et visualisations.
- Pour l'entraînement et l'analyse complets, utiliser les scripts dans `src/pipelines/`.
- Les notebooks supposent que les runs existent dans le dossier `runs/`.
