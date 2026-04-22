from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CURVE_METRICS = [
    "val_loss",
    "val_accuracy",
    "global_beta_sq_mean",
    "global_low_frequency_ratio_mean",
    "global_total_energy_mean",
    "global_sigma_energy_ratio_mean",
    "global_grad_x_energy_ratio_mean",
    "global_grad_y_energy_ratio_mean",
    "global_grad_xy_energy_ratio_mean",
]

DEFAULT_PRETTY_NAMES = {
    "val_loss": "Validation loss",
    "val_accuracy": "Validation accuracy",
    "global_beta_sq_mean": "Beta squared",
    "global_low_frequency_ratio_mean": "Low-frequency ratio",
    "global_total_energy_mean": "Total energy",
    "global_sigma_energy_ratio_mean": "Sigma energy ratio",
    "global_grad_x_energy_ratio_mean": "Grad X energy ratio",
    "global_grad_y_energy_ratio_mean": "Grad Y energy ratio",
    "global_grad_xy_energy_ratio_mean": "Grad XY energy ratio",
}


def build_curve_dataset(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et prépare le dataset inter-modèle pour les courbes.

    Sortie:
        1 ligne = 1 couple (init, seed, epoch)
    """
    required_cols = [
        "init",
        "seed",
        "epoch",
        "val_loss",
        "val_accuracy",
        "global_beta_sq_mean",
        "global_low_frequency_ratio_mean",
        "global_total_energy_mean",
        "global_sigma_energy_ratio_mean",
        "global_grad_x_energy_ratio_mean",
        "global_grad_y_energy_ratio_mean",
        "global_grad_xy_energy_ratio_mean",
    ]

    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing columns in df_all for curves: {missing}")

    df_curve = df_all[required_cols].copy()

    # Une seule ligne par (init, seed, epoch)
    df_curve = df_curve.drop_duplicates(subset=["init", "seed", "epoch"])

    # Tri propre
    df_curve = df_curve.sort_values(["init", "seed", "epoch"]).reset_index(drop=True)

    if df_curve.empty:
        raise ValueError("Empty curve dataset after cleaning.")

    return df_curve


def aggregate_curve_metrics(df_curve: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les seeds pour obtenir une courbe moyenne par initialisation.
    """
    required_cols = {"init", "epoch"}
    if not required_cols.issubset(df_curve.columns):
        raise ValueError(f"df_curve must contain columns {sorted(required_cols)}")

    df_mean = (
        df_curve
        .groupby(["init", "epoch"])
        .mean(numeric_only=True)
        .reset_index()
    )

    return df_mean


def plot_metric(
    df_mean: pd.DataFrame,
    metric: str,
    output_dir: Path,
    pretty_names: Optional[dict[str, str]] = None,
) -> Path:
    """
    Génère et sauvegarde une courbe inter-modèle pour une métrique.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metric not in df_mean.columns:
        raise ValueError(f"Metric '{metric}' not found in df_mean.")

    pretty_names = pretty_names or DEFAULT_PRETTY_NAMES
    title = pretty_names.get(metric, metric)

    plt.figure(figsize=(8, 5))

    for init_name in sorted(df_mean["init"].unique()):
        sub = df_mean[df_mean["init"] == init_name].sort_values("epoch")
        plt.plot(sub["epoch"], sub[metric], marker="o", label=init_name)

    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(f"{title} vs epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = output_dir / f"{metric}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_all_metrics(
    df_mean: pd.DataFrame,
    output_dir: Path,
    metrics: Optional[list[str]] = None,
    pretty_names: Optional[dict[str, str]] = None,
) -> None:
    """
    Génère toutes les courbes inter-modèle demandées.
    """
    metrics = metrics or DEFAULT_CURVE_METRICS
    pretty_names = pretty_names or DEFAULT_PRETTY_NAMES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [m for m in metrics if m not in df_mean.columns]
    if missing:
        raise ValueError(f"Missing metrics in df_mean: {missing}")

    for metric in metrics:
        plot_metric(
            df_mean=df_mean,
            metric=metric,
            output_dir=output_dir,
            pretty_names=pretty_names,
        )