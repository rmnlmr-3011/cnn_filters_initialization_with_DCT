# SYS809 — Phase 2 : DCT et analyse locale

## Objectif
À partir des filtres 3x3 d’un modèle entraîné, projeter les noyaux dans la base DCT et calculer des métriques locales inspirées du papier de référence.

## Unité d’analyse
L’unité d’analyse élémentaire est un noyau spatial 3x3 associé à une paire `(in_channel, out_channel)`.

Justification :
dans TensorFlow / Keras, les poids d’une couche `Conv2D` ont la forme :

`(kernel_height, kernel_width, in_channels, out_channels)`

Donc pour une couche 3x3, chaque paire `(in_channel, out_channel)` correspond à un noyau 3x3 distinct.

## Métriques attendues
Pour chaque noyau 3x3, on calcule :

- les coefficients DCT
- l’énergie par composante DCT
- l’énergie totale
- le ratio de basses fréquences
- beta

## Niveaux d’agrégation
Les résultats doivent pouvoir être agrégés à trois niveaux :

- par noyau
- par filtre de sortie
- par couche

## Remarque
La phase 2 est une phase d’analyse offline.
Elle ne modifie ni l’architecture du modèle, ni l’entraînement, ni les hyperparamètres.


# model
#   ↓
# get_conv3x3_layers
#   ↓
# get_layer_filters
#   ↓
# iter_conv_kernels
#   ↓
# kernel 3x3
#   ↓
# DCT → metrics → analyse