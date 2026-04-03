# Point d'entrée pour analyser un modèle Keras déjà entraîné
# Charge le modèle, lance l'analyse des filtres, puis sauvegarde :
# - kernel_metrics.csv
# - filter_metrics.csv
# - layer_metrics.csv
# - filter_analysis_summary.json


# Pour éxécuter depuis terminal avec modèle de test :
# python -m src.analysis.run_filter_analysis --model_path runs/baseline_resnet20_cifar10_he/model.keras --output_dir runs/baseline_resnet20_cifar10_he/analysis


from pathlib import Path
import argparse
import tensorflow as tf

from src.analysis.filter_analysis import analyze_model
from src.utils.io import save_json
from src.utils.logging import get_logger


def main():
    parser = argparse.ArgumentParser(description="Run filter analysis on a trained Keras model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.keras)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where analysis outputs will be saved. Defaults to model parent directory."
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_path.suffix != ".keras":
        raise ValueError(f"Expected a .keras model file, got: {model_path}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("filter_analysis")

    logger.info("===================================")
    logger.info("Filter analysis configuration")
    logger.info("===================================")
    logger.info(f"model_path: {model_path}")
    logger.info(f"output_dir: {output_dir}")
    logger.info("===================================")

    logger.info("Loading model...")
    model = tf.keras.models.load_model(model_path)

    logger.info("Running filter analysis...")
    kernel_df, filter_df, layer_df, summary = analyze_model(model)

    kernel_csv_path = output_dir / "kernel_metrics.csv"
    filter_csv_path = output_dir / "filter_metrics.csv"
    layer_csv_path = output_dir / "layer_metrics.csv"
    summary_json_path = output_dir / "filter_analysis_summary.json"

    logger.info("Saving outputs...")
    kernel_df.to_csv(kernel_csv_path, index=False)
    filter_df.to_csv(filter_csv_path, index=False)
    layer_df.to_csv(layer_csv_path, index=False)
    save_json(summary, summary_json_path)

    logger.info(f"kernel_metrics.csv saved to: {kernel_csv_path}")
    logger.info(f"filter_metrics.csv saved to: {filter_csv_path}")
    logger.info(f"layer_metrics.csv saved to: {layer_csv_path}")
    logger.info(f"filter_analysis_summary.json saved to: {summary_json_path}")
    logger.info("Filter analysis completed successfully.")


if __name__ == "__main__":
    main()