# Code exécutable pour analyser tous les checkpoints d'un run donné. Les résultats sont enregistrés dans analysis/checkpoints/ et analysis/final/ si include_final_model est activé.

# python -m src.analysis.run_checkpoint_analysis --run_dir runs/baseline_resnet20_cifar10_he_20260406_170145 --include_final_model


from pathlib import Path
import argparse

from src.analysis.checkpoint_analysis import analyze_run_checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Analyze all checkpoint models from a run directory."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the run directory containing checkpoints/ and model.keras"
    )
    parser.add_argument(
        "--include_final_model",
        action="store_true",
        help="Include run_dir/model.keras and save its analysis to analysis/final"
    )
    parser.add_argument(
        "--include_epoch_final",
        action="store_true",
        help="Include checkpoints/epoch_final.keras"
    )

    args = parser.parse_args()

    analyze_run_checkpoints(
        run_dir=Path(args.run_dir),
        include_final_model=args.include_final_model,
        include_epoch_final=args.include_epoch_final,
    )


if __name__ == "__main__":
    main()