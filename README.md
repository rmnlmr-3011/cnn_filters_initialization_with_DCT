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

Arborescence :

cnn-dct-init/
├── README.md
├── .gitignore
├── environment.yml
├── configs/
│   ├── base.yaml
│   ├── init_he.yaml
│   ├── init_sigma.yaml
│   ├── init_grad.yaml
│   ├── init_dctlow.yaml
│   └── init_dctlow_noise.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loaders.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet20.py
│   │   └── initializers.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── callbacks.py
│   │   ├── evaluate.py
│   │   └── train.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── dct.py
│   │   ├── filter_metrics.py
│   │   └── plots.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── io.py
│       ├── logging.py
│       └── seed.py
├── notebooks/
│   ├── 01_sanity_checks.ipynb
│   ├── 02_filter_analysis.ipynb
│   └── 03_results_figures.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── runs/
└── tests/
    ├── test_dct.py
    ├── test_initializers.py
    ├── test_metrics.py
    └── test_resnet20.py