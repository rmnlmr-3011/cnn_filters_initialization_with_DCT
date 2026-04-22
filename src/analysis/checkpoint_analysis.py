# Code pour analyser tous les checkpoints d'un run, et sauvegarder les résultats

from pathlib import Path
from typing import List, Tuple, Union

import tensorflow as tf

from src.analysis.filter_analysis import analyze_model
from src.utils.io import save_json
from src.utils.logging import get_logger

from src.models.initializers import (
    SigmaPureInitializer,
    SigmaInitializer,
    GradVerticalPureInitializer,
    GradHorizontalPureInitializer,
    GradInitializer,
    DCTLowInitializer,
    DCTLowNoiseInitializer,
)

CUSTOM_OBJECTS = {
    "SigmaPureInitializer": SigmaPureInitializer,
    "SigmaInitializer": SigmaInitializer,
    "GradVerticalPureInitializer": GradVerticalPureInitializer,
    "GradHorizontalPureInitializer": GradHorizontalPureInitializer,
    "GradInitializer": GradInitializer,
    "DCTLowInitializer": DCTLowInitializer,
    "DCTLowNoiseInitializer": DCTLowNoiseInitializer,
}


def list_checkpoint_models(run_dir: Path) -> List[Tuple[str, Path]]:
    """
    Retourne la liste des modèles à analyser sous la forme :
    [
        ("epoch_000", Path(...)),
        ("epoch_001", Path(...)),
        ...
        ("final", Path(...)),
    ]
    """
    run_dir = Path(run_dir)
    checkpoints_dir = run_dir / "checkpoints"

    items: List[Tuple[str, Path]] = []

    if checkpoints_dir.exists():
        checkpoint_paths = sorted(checkpoints_dir.glob("epoch_*.keras"))
        for path in checkpoint_paths:
            label = path.stem  # ex: epoch_000, epoch_001, epoch_final
            items.append((label, path))

    final_model_path = run_dir / "model.keras"
    if final_model_path.exists():
        items.append(("final", final_model_path))

    return items


def analyze_single_model(model_path: Path, output_dir: Path, logger=None) -> None:
    """
    Charge un modèle .keras, lance l'analyse phase 2,
    puis sauvegarde les 4 sorties standard.
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if logger is not None:
        logger.info(f"Loading model: {model_path}")

    model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    if logger is not None:
        logger.info(f"Running filter analysis for: {model_path.name}")

    kernel_df, filter_df, layer_df, summary = analyze_model(model)

    kernel_df.to_csv(output_dir / "kernel_metrics.csv", index=False)
    filter_df.to_csv(output_dir / "filter_metrics.csv", index=False)
    layer_df.to_csv(output_dir / "layer_metrics.csv", index=False)
    save_json(summary, output_dir / "filter_analysis_summary.json")

    if logger is not None:
        logger.info(f"Analysis saved to: {output_dir}")


def analyze_run_checkpoints(
    run_dir: Union[Path, str],
    include_final_model: bool = True,
    include_epoch_final: bool = False,
) -> None:
    """
    Analyse automatiquement tous les checkpoints d'un run.

    Paramètres
    ----------
    run_dir:
        Dossier du run.
    include_final_model:
        Si True, analyse aussi run_dir/model.keras vers analysis/final.
    include_epoch_final:
        Si False, ignore checkpoints/epoch_final.keras pour éviter
        un doublon avec model.keras.
    """
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("checkpoint_analysis")

    logger.info("===================================")
    logger.info("Checkpoint series analysis")
    logger.info("===================================")
    logger.info(f"run_dir: {run_dir}")
    logger.info(f"analysis_dir: {analysis_dir}")
    logger.info("===================================")

    items = list_checkpoint_models(run_dir)

    if not include_final_model:
        items = [(label, path) for label, path in items if label != "final"]

    if not include_epoch_final:
        items = [(label, path) for label, path in items if label != "epoch_final"]

    if len(items) == 0:
        raise FileNotFoundError(
            f"No checkpoint models found in run directory: {run_dir}"
        )

    for label, model_path in items:
        output_dir = analysis_dir / label
        analyze_single_model(model_path=model_path, output_dir=output_dir, logger=logger)

    logger.info("All checkpoint analyses completed successfully.")