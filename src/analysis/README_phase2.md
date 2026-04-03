# SYS809 — Phase 2 : DCT et analyse locale

## Objectif
À partir des filtres 3x3 d’un modèle entraîné, projeter les noyaux dans la base DCT et calculer des métriques locales inspirées du papier de référence.

Cette phase vise à construire une **brique d’analyse offline fiable, modulaire et testée**, permettant d’extraire des informations structurelles sur les filtres appris.

---

## Unité d’analyse

L’unité d’analyse élémentaire est un noyau spatial 3x3 associé à une paire `(in_channel, out_channel)`.

Justification :  
dans TensorFlow / Keras, les poids d’une couche `Conv2D` ont la forme :


Donc pour une couche 3x3, chaque paire `(in_channel, out_channel)` correspond à un noyau 3x3 distinct.

---

## Métriques implémentées

Pour chaque noyau 3x3, on calcule :

### Bloc A — métriques DCT
- coefficients DCT
- énergie par composante DCT
- énergie totale
- ratio de basses fréquences (Σ + ∇x + ∇y)

### Bloc B — symétrie / antisymétrie
- décomposition symétrique / antisymétrique
- `beta²` : ratio d’énergie antisymétrique sur énergie totale

---

## Niveaux d’agrégation

Les métriques sont disponibles à trois niveaux :

- **niveau noyau** : `(in_channel, out_channel)`
- **niveau filtre** : agrégation sur les `in_channel` pour un `out_channel`
- **niveau couche** : agrégation sur tous les noyaux de la couche

---

## Architecture du code

La phase 2 est organisée en modules indépendants :

### DCT
- `dct.py`  
  → implémente la DCT 2D, l’IDCT et la projection des filtres

- `dct_bases.py`  
  → définit la base DCT 3x3, les indices et les masques (Σ, ∇x, ∇y)

### Extraction des filtres
- `model_filters.py`  
  → extrait les couches `Conv2D 3x3` et itère sur les noyaux

### Métriques
- `filter_metrics.py`  
  → calcule les métriques DCT et sym/antisym  
  → contient également les fonctions d’agrégation (noyau / filtre / couche)

### Analyse globale
- `filter_analysis.py`  
  → logique métier d’analyse d’un modèle Keras déjà chargé  
  → produit :
  - métriques noyau
  - métriques filtre
  - métriques couche
  - résumé global

### Script exécutable
- `run_filter_analysis.py`  
  → point d’entrée CLI  
  → charge un modèle `.keras`  
  → lance l’analyse  
  → sauvegarde les résultats

---

## Pipeline d’analyse
model
↓
get_conv3x3_layers
↓
iter_conv_kernels
↓
kernel 3x3
↓
DCT → métriques → agrégations
↓
DataFrames (kernel / filter / layer)
↓
export CSV + JSON


---

## Sorties générées

Pour un modèle donné, l’analyse produit :

- `kernel_metrics.csv`  
  → métriques pour chaque noyau

- `filter_metrics.csv`  
  → métriques agrégées par filtre (`out_channel`)

- `layer_metrics.csv`  
  → métriques agrégées par couche

- `filter_analysis_summary.json`  
  → résumé global (statistiques agrégées)

---

## Tests

Deux niveaux de validation ont été mis en place :

### Notebook
- `02_filter_analysis.ipynb`
→ validation exploratoire des étapes 2 à 5

### Tests automatisés (pytest)
- `tests/test_dct.py`
- `tests/test_filter_metrics.py`

Tests réalisés :
- reconstruction DCT / IDCT
- identité de Parseval
- cohérence des projections
- validité des ratios basse fréquence
- validité de `beta²`
- cohérence des agrégations

---

## Portée de la phase 2

- phase **entièrement offline**
- aucune modification :
  - de l’architecture
  - de l’entraînement
  - des hyperparamètres

Cette phase fournit une **brique analytique réutilisable** pour les phases suivantes.

---

## Limites actuelles

- statistiques limitées à moyenne / écart-type (extensible)
- pas d’analyse d’orientation (considérée comme extension future)
- analyse effectuée post-entraînement uniquement

---

## Extensions possibles

- ajout de métriques d’orientation (phase d’analyse scientifique)
- enrichissement des statistiques (min, max, percentiles)
- visualisations avancées (histogrammes, heatmaps DCT)
- analyse temporelle (évolution des filtres pendant l’entraînement)

---

---

## Sorties générées

Pour un modèle donné, l’analyse produit :

- `kernel_metrics.csv`  
  → métriques pour chaque noyau

- `filter_metrics.csv`  
  → métriques agrégées par filtre (`out_channel`)

- `layer_metrics.csv`  
  → métriques agrégées par couche

- `filter_analysis_summary.json`  
  → résumé global (statistiques agrégées)

---

## Tests

Deux niveaux de validation ont été mis en place :

### Notebook
- `02_filter_analysis.ipynb`
→ validation exploratoire des étapes 2 à 5

### Tests automatisés (pytest)
- `tests/test_dct.py`
- `tests/test_filter_metrics.py`

Tests réalisés :
- reconstruction DCT / IDCT
- identité de Parseval
- cohérence des projections
- validité des ratios basse fréquence
- validité de `beta²`
- cohérence des agrégations

---

## Portée de la phase 2

- phase **entièrement offline**
- aucune modification :
  - de l’architecture
  - de l’entraînement
  - des hyperparamètres

Cette phase fournit une **brique analytique réutilisable** pour les phases suivantes.

---

## Limites actuelles

- statistiques limitées à moyenne / écart-type (extensible)
- pas d’analyse d’orientation (considérée comme extension future)
- analyse effectuée post-entraînement uniquement

---

## Extensions possibles

- ajout de métriques d’orientation (phase d’analyse scientifique)
- enrichissement des statistiques (min, max, percentiles)
- visualisations avancées (histogrammes, heatmaps DCT)
- analyse temporelle (évolution des filtres pendant l’entraînement)

---

## Arborescence

analysis/
┣ dct.py
┣ dct_bases.py
┣ filter_analysis.py
┣ filter_metrics.py
┣ model_filters.py
┣ README_phase2.md
┗ run_filter_analysis.py## Arborescence


















 Ancien ReadME :

 Fonctions:
1. dct2 et idct2
Fonctions pour la DCT 2D et l’inverse. 
Elles permettent de :
analyser les filtres appris en fréquence ;
reconstruire des filtres 3×3 à partir de composantes fréquentielles choisies.

2. get_filter(model, layer, sev=False)
Cette fonction parcourt les couches du modèle, garde les convolutions, puis renvoie les poids d’une couche choisie. Elle filtre en priorité les convolutions 3x3, et peut aussi inclure du 7x7 si sev=True. 

3. getSymAntiSymTF(filter)
Cette fonction sépare un noyau en partie symétrique et antisymétrique à partir de flips horizontaux, verticaux et rotation. 

4. getSobelTF(f)
Elle calcule une orientation locale à partir d’un noyau de type Sobel construit avec getDerivKernels. C’est une brique intermédiaire utilisée par getDominantAngle.

5. getDominantAngle(filters)
Cette fonction combine la partie antisymétrique et l’orientation Sobel pour donner un angle dominant par filtre.

6. topKfilters et topKchannels
Ces fonctions classent les filtres ou les canaux par magnitude.Fonctions utiles pour visualiser les filtres les plus forts.

