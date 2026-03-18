Ce projet répond à la question suivante : "Ce projet répond à la question suivante : “Quel est l’impact d’une initialisation structurée DCT-low des filtres 3×3 sur la convergence, la stabilité, la performance finale et l’évolution fréquentielle des filtres, en comparaison à l’initialisation He ?”

La structure du code :

- configs/ contient les fichiers de configuration des expériences

- src/data/ contient le dataset CIFAR10

- src/models/ contient l'architecture du modèle Resnet20 ainsi que les initialisations custom

- src/training/ contient la boucle d'entraînement, l'évaluation et les callbacks

- src/analysis/ contient les options de visualisation et d'évaluation des composantes DCT

- src/utils/ contient quelques fonctions annexes

- notebooks/ contient des .ipynb pour tester rapidement

- runs/ contient tous les résultats des entraînements 

- tests/ contient tous les résultats des tests