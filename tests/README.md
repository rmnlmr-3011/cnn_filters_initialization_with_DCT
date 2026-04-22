# Tests

Ce dossier contient les tests unitaires du projet, exécutés avec `pytest`.

## Objectif

Les tests permettent de vérifier le bon fonctionnement des fonctions de base, des métriques DCT et des initialisateurs personnalisés qui servent à l’entraînement des modèles.

## Comment exécuter

Depuis la racine du projet :

```bash
pytest tests
```

## Contenu des fichiers

- `test_dct.py`
  - Vérifie la reconstruction DCT/IDCT des kernels 3×3.
  - S’assure que l’énergie est conservée entre les domaines spatial et DCT.
  - Contrôle la forme et la cohérence de la projection des filtres 4D vers le domaine DCT.

- `test_filter_metrics.py`
  - Vérifie le calcul des coefficients DCT et de l’énergie DCT.
  - Teste le ratio d’énergie basse fréquence.
  - Vérifie la décomposition symétrique / antisymétrique des kernels.
  - Vérifie le calcul de la métrique `beta_sq`.
  - Contrôle la génération du résumé de métriques d’un kernel.

- `test_initializers.py`
  - Vérifie que `get_initializer()` retourne des initialisateurs valides.
  - Vérifie les formes de poids générées pour plusieurs initialisateurs : `he`, `sigma_pure`, `sigma`, `grad`, `grad_vertical_pure`, `grad_horizontal_pure`, `dctlow`, `dctlow_noise`.
  - Vérifie la reproductibilité des initialisations avec la même graine.
  - Vérifie le comportement de fallback pour des formes non 3×3.
  - Vérifie les propriétés spécifiques de certains initialisateurs : noyau constant, énergie basse fréquence, kernels de type Sobel.
  - Vérifie l’intégration des initialisateurs dans un modèle `ResNet20` et la compatibilité d’une passe avant.
  - Vérifie l’assignation granulaire d’initialisateurs à différentes couches du ResNet20.

## Remarque

`__init__.py` transforme ce dossier en package Python, mais les tests sont lancés via `pytest` et ne nécessitent pas d’import spécial.
