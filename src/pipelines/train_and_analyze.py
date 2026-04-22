# Code permettant d'exécuter la pipeline complète : entraînement du modèle, analyse des checkpoints, et construction de la série temporelle des métriques d'analyse de filtres.

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# & "C:\Users\rmnlm\anaconda3\shell\condabin\conda-hook.ps1"
# conda activate sys809

# cd cnn_filters_initialization_with_DCT
# python -m src.pipelines.train_and_analyze

# python -m src.pipelines.train_and_analyze --config path/to/your.yaml



# "configs/init_sigma_pure.yaml"




import sys
from pathlib import Path
import argparse

from src.training.train_core import run_training
from src.analysis.checkpoint_analysis import analyze_run_checkpoints
from src.analysis.checkpoint_series import build_checkpoint_series
from src.utils.logging import get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Train a model, analyze checkpoints, and build checkpoint series summary."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the experiment config file."
    )
    parser.add_argument(
        "--include_final_model",
        action="store_true",
        help="Include run_dir/model.keras in checkpoint analysis."
    )
    parser.add_argument(
        "--include_epoch_final",
        action="store_true",
        help="Include checkpoints/epoch_final.keras in checkpoint analysis."
    )

    args = parser.parse_args()

    logger = get_logger("train_and_analyze")

    logger.info("===================================")
    logger.info("Full pipeline: train + checkpoint analysis + series summary")
    logger.info("===================================")
    logger.info(f"config: {args.config}")
    logger.info("===================================")

    # Étape 1 : entraînement
    run_dir = run_training(config_path=args.config)

    logger.info(f"Training completed. Run directory: {run_dir}")

    # Étape 2 : analyse des checkpoints
    analyze_run_checkpoints(
        run_dir=run_dir,
        include_final_model=args.include_final_model,
        include_epoch_final=args.include_epoch_final,
    )

    logger.info("Checkpoint analysis completed.")

    # Étape 3 : analyse inter-checkpoints
    build_checkpoint_series(run_dir)

    logger.info("Checkpoint series summary completed.")
    logger.info("Full pipeline completed successfully.")


if __name__ == "__main__":
    main()