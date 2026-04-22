# configs/

Ce dossier contient les fichiers YAML de configuration pour les différentes expériences d'entraînement et d'initialisation. Chaque fichier définit complètement les paramètres d'une exécution.

## Contention des fichiers

Les fichiers suivent le pattern : `init_<initializer_name>_<seed>.yaml`

Exemple : `init_dctlow_42.yaml`, `init_sigma_pure_43.yaml`, etc.

## Structure d'un fichier YAML

Voici un exemple annoté de la structure générale :

```yaml
# ================================================================
# 1. EXPÉRIENCE
# ================================================================
experiment_name: init_resnet20_cifar10_he_seed42
seed: 42

# ================================================================
# 2. DATASET
# ================================================================
dataset:
  name: cifar10                    # Nom du dataset à charger
  validation_split: 0.1            # Ratio train/validation (10%)
  normalize: true                  # Normalisation des données
  augment: false                   # Augmentation de données (désactivée)

# ================================================================
# 3. MODÈLE
# ================================================================
model:
  name: resnet20                   # Architecture utilisée
  input_shape: [32, 32, 3]         # Forme de l'entrée (32×32 RGB)
  num_classes: 10                  # Nombre de classes (CIFAR-10)

# ================================================================
# 4. INITIALISEUR
# ================================================================
initializer:
  mode: global                     # Mode global ou per_layer
  name: he                         # Nom de l'initialiseur
  scale: 1.0                       # Facteur d'échelle
  noise_std: 0.0                   # Écart-type du bruit additionnel
  seed: 42                         # Graine de reproductibilité

# ================================================================
# 5. ENTRAÎNEMENT
# ================================================================
training:
  optimizer: sgd                   # Optimiseur (SGD)
  learning_rate: 0.05             # Taux d'apprentissage initial
  momentum: 0.9                    # Momentum SGD
  batch_size: 32                   # Taille du batch
  epochs: 20                       # Nombre d'époque

  scheduler:                       # Planification du learning rate
    type: multistep               # Type : descente par étapes
    milestones: [8, 16]           # Étapes où réduire (après époque 8, 16)
    gamma: 0.1                    # Facteur de réduction (× 0.1)
  
  early_stopping:                  # Arrêt anticipé
    enabled: false                # Désactivé
    monitor: val_loss             # Métrique à surveiller
    patience: 5                   # Nombre d'époque sans amélioration
    min_delta: 0.0                # Amélioration minimale requise
    restore_best_weights: true    # Restaurer les meilleurs poids

# ================================================================
# 6. RUNTIME
# ================================================================
runtime:
  run_root: runs                   # Dossier racine des résultats
  deterministic: true              # Mode déterministe (reproductibilité)
  verbose: 1                       # Niveau de verbosité (0-2)

# ================================================================
# 7. CHECKPOINTS
# ================================================================
checkpoints:
  enabled: true                    # Activation de la sauvegarde
  save_initial: true              # Sauvegarder modèle initial
  save_final: true                # Sauvegarder modèle final
  save_epochs: [1, 2, 3, ..., 20]  # Sauvegarder après chaque époque
```

## Sections détaillées

### experiment_name
Identifiant unique pour l'expérience. Format recommandé : `init_<model>_<dataset>_<init>_seed<seed>`.

### seed
Graine aléatoire pour assurer la reproductibilité. Trois seeds courantes : `42`, `43`, `44`.

### dataset
- `name` : Dataset à utiliser (`cifar10`).
- `validation_split` : Ratio train/validation (généralement 0.1).
- `normalize` : Normaliser les pixels en [0, 1].
- `augment` : Appliquer l'augmentation de données (généralement désactivée).

### model
- `name` : Architecture du modèle (`resnet20`).
- `input_shape` : Dimensions d'entrée `[hauteur, largeur, canaux]`.
- `num_classes` : Nombre de classes de sortie.

### initializer
- `mode` : `global` (initialiseur global) ou `per_layer` (différent par couche).
- `name` : Initialiseur parmi : `he`, `sigma`, `sigma_pure`, `grad`, `grad_vertical_pure`, `grad_horizontal_pure`, `dctlow`, `dctlow_noise`.
- `scale` : Facteur d'échelle multiplicatif (généralement 1.0).
- `noise_std` : Bruit gaussien additif (pour les variantes bruitées).
- `seed` : Graine pour l'initialiseur.

### training
- `optimizer` : `sgd` avec `learning_rate` et `momentum`.
- `batch_size` : Taille du batch (généralement 32).
- `epochs` : Nombre d'époque (généralement 20).
- `scheduler` : Réduction du learning rate par étapes (`multistep`).
- `early_stopping` : Critère d'arrêt anticipé (généralement désactivé).

### runtime
- `run_root` : Répertoire de sortie pour les runs.
- `deterministic` : Mode déterministe pour la reproductibilité.
- `verbose` : Niveau de détail du logging.

### checkpoints
- `enabled` : Activer la sauvegarde des modèles.
- `save_initial` : Sauvegarder le modèle initial.
- `save_final` : Sauvegarder le modèle final.
- `save_epochs` : Numéros des époque à sauvegarder (généralement toutes).

## Initialiseurs disponibles

Le dossier contient des fichiers pour les initialiseurs suivants (avec seeds 42, 43, 44) :

- `he` : Initialisation He (baseline classique).
- `sigma` : Initialisation sigma stochastique (avec bruit).
- `sigma_pure` : Initialisation sigma déterministe.
- `grad` : Initialisation gradient stochastique.
- `grad_vertical_pure` : Kernel Sobel vertical pur.
- `grad_horizontal_pure` : Kernel Sobel horizontal pur.
- `dctlow` : DCT basse fréquence déterministe.
- `dctlow_noise` : DCT basse fréquence avec bruit.

## Création d'une nouvelle configuration

Pour créer une nouvelle configuration, copier un fichier existant et modifier les champs pertinents :

```bash
cp init_he_42.yaml init_mon_init_45.yaml
# Éditer init_mon_init_45.yaml et modifier :
# - experiment_name
# - seed
# - initializer.name
```

## Notes

- Les fichiers YAML sont lus par `src/utils/config.py`.
- Tous les fichiers du dossier `configs/` ont été testés et analysés.
- Voir aussi `configs_hybrid/` pour des expériences complexes non testées.
