from pathlib import Path
import argparse
import sys

from src.training.train_core import run_training
from src.analysis.checkpoint_analysis import analyze_run_checkpoints
from src.analysis.checkpoint_series import build_checkpoint_series
from src.utils.logging import get_logger

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# & "C:\Users\rmnlm\anaconda3\shell\condabin\conda-hook.ps1"
# conda activate sys809

# python -m src.pipelines.train_and_analyze_batch --config_dir configs/ --continue_on_error

def run_full_pipeline_for_config(
    config_path: Path,
    include_final_model: bool = False,
    include_epoch_final: bool = False,
) -> Path:
    """
    Exécute la pipeline complète pour un seul fichier de config.
    """
    logger = get_logger("train_and_analyze_batch")

    logger.info("===================================")
    logger.info(f"Running full pipeline for config: {config_path}")
    logger.info("===================================")

    # Étape 1 : entraînement
    run_dir = run_training(config_path=config_path)
    logger.info(f"Training completed. Run directory: {run_dir}")

    # Étape 2 : analyse des checkpoints
    analyze_run_checkpoints(
        run_dir=run_dir,
        include_final_model=include_final_model,
        include_epoch_final=include_epoch_final,
    )
    logger.info("Checkpoint analysis completed.")

    # Étape 3 : analyse inter-checkpoints
    build_checkpoint_series(run_dir)
    logger.info("Checkpoint series summary completed.")

    return run_dir


def collect_config_paths(config_dir: Path) -> list[Path]:
    """
    Récupère tous les fichiers YAML d'un dossier, triés par nom.
    """
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    config_paths = sorted(config_dir.glob("*.yaml"))
    if len(config_paths) == 0:
        raise FileNotFoundError(f"No .yaml files found in: {config_dir}")

    return config_paths


def main():
    parser = argparse.ArgumentParser(
        description="Run train+analyze sequentially for all YAML configs in a directory."
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        required=True,
        help="Directory containing YAML experiment configs."
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
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="If set, continue with next config when one run fails."
    )

    args = parser.parse_args()

    logger = get_logger("train_and_analyze_batch")

    config_dir = Path(args.config_dir)
    config_paths = collect_config_paths(config_dir)

    logger.info("===================================")
    logger.info("Batch pipeline: train + checkpoint analysis + series summary")
    logger.info("===================================")
    logger.info(f"config_dir: {config_dir}")
    logger.info(f"n_configs: {len(config_paths)}")
    logger.info("Configs to run:")
    for path in config_paths:
        logger.info(f" - {path.name}")
    logger.info("===================================")

    successes = []
    failures = []

    for idx, config_path in enumerate(config_paths, start=1):
        logger.info("###################################")
        logger.info(f"[{idx}/{len(config_paths)}] Processing: {config_path.name}")
        logger.info("###################################")

        try:
            run_dir = run_full_pipeline_for_config(
                config_path=config_path,
                include_final_model=args.include_final_model,
                include_epoch_final=args.include_epoch_final,
            )
            successes.append((config_path, run_dir))
            logger.info(f"[SUCCESS] {config_path.name} -> {run_dir}")

        except Exception as e:
            failures.append((config_path, str(e)))
            logger.exception(f"[FAILURE] {config_path.name}: {e}")

            if not args.continue_on_error:
                logger.error("Stopping batch because --continue_on_error was not set.")
                break

    logger.info("===================================")
    logger.info("Batch summary")
    logger.info("===================================")
    logger.info(f"Successes: {len(successes)}")
    logger.info(f"Failures: {len(failures)}")

    if successes:
        logger.info("Successful runs:")
        for config_path, run_dir in successes:
            logger.info(f" - {config_path.name} -> {run_dir}")

    if failures:
        logger.info("Failed runs:")
        for config_path, error_msg in failures:
            logger.info(f" - {config_path.name}: {error_msg}")

    logger.info("Batch pipeline finished.")

    # Code retour utile si tu veux détecter un échec dans un script shell
    if failures and not args.continue_on_error:
        sys.exit(1)


if __name__ == "__main__":
    main()