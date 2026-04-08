# SYS809 — Phase 5




epoch_000.keras = état initial avant entraînement
epoch_001.keras = état après 1 epoch terminé
epoch_005.keras = état après 5 epochs terminés
epoch_final.keras = état final




## 1. Quelles couches sauvegarder?

- Couches Conv2D 3x3, pas 1x1

## 2. Que sauvegarder dans ces couches?

- Uniquement les poids bruts?
=> Puis recalculer en aval les métriques phase 2

## 3. A quels instants sauvegarder?

- Ni tous, ni 1 seul
- Il faut pouvoir le configurer (option du yaml)
  - Idée 1 : 0, 1, tous les multiples de N, puis final
    Exempke : 0, 1, 5, 10, 15, 18 (final)

## 4. A quel niveau de granularité logique sauvegarder?

- Au niveau couche?




Questions au chat :







Question générale : à quoi permettent les snapshots comparés à ce qui a été fait à l'étape 2? En pratique on va entrainer le modèle et les callbacks de la phase 5 enregistreront certaines données. Parmi ces données, lesquelles proviennent de ce qui a établi à la phase 2 et lesquelles devront être traitées dans aval avec les fonctions de la phase 2?











Mais j'ai déjà créer des callbacks lors de la phase 1. En quoi ces derniers diffèrent-ils? (je t'ai joint les fichiers produits par les callbacks de la phase 1) (JOINDRE CALLBACKS DE PHASE 1). Après mes callbacks phase 1 n'enregistre le modèle qu'à la fin de l'entrainement alors que la ce serait tout au long de l'entrainement, c'est ca l'unique différence?


Mais dans le fond ne suffirait-il juste pas de sauvegarder le modèle à chaque epoch choisi (0, 1, 5, 10...). Et pour la post-analyse, repasser ce modèle dans les fonctions de la phase 2?