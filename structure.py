from pathlib import Path

root = Path("cnn_filters_initialization_with_DCT")

dirs = [
    "configs",
    "src/data",
    "src/models",
    "src/training",
    "src/analysis",
    "src/utils",
    "notebooks",
    "reports/figures",
    "reports/tables",
    "runs",
    "tests",
]

files = [
    "README.md",
    "environment.yml",
    ".gitignore",
    "configs/base.yaml",
    "configs/init_he.yaml",
    "configs/init_sigma.yaml",
    "configs/init_grad.yaml",
    "configs/init_dctlow.yaml",
    "src/data/loaders.py",
    "src/models/cnn.py",
    "src/models/initializers.py",
    "src/training/train.py",
    "src/training/evaluate.py",
    "src/training/callbacks.py",
    "src/analysis/dct.py",
    "src/analysis/filter_metrics.py",
    "src/analysis/plots.py",
    "src/analysis/pretrained_patterns.py",
    "src/utils/seed.py",
    "src/utils/io.py",
    "src/utils/logging.py",
    "notebooks/01_sanity_checks.ipynb",
    "notebooks/02_filter_analysis.ipynb",
    "notebooks/03_results_figures.ipynb",
    "reports/draft_report.md",
    "tests/test_dct.py",
    "tests/test_initializers.py",
    "tests/test_metrics.py",
]

for d in dirs:
    (root / d).mkdir(parents=True, exist_ok=True)

for f in files:
    path = root / f
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)