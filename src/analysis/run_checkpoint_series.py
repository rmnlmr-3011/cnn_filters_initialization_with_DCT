# Code exécutable pour construire une série temporelle des métriques d'analyse de filtres pour tous les checkpoints d'un run. Les résultats sont enregistrés dans analysis/checkpoint_series.csv et analysis/checkpoint_series.png.

# python -m src.analysis.run_checkpoint_series --run_dir "runs\baseline_resnet20_cifar10_he_20260406_170145"

from pathlib import Path
import argparse

from src.analysis.checkpoint_series import build_checkpoint_series


def main():
    parser = argparse.ArgumentParser(
        description="Build a checkpoint series summary and plots from analysis outputs."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the run directory containing analysis/"
    )

    args = parser.parse_args()

    build_checkpoint_series(Path(args.run_dir))


if __name__ == "__main__":
    main()