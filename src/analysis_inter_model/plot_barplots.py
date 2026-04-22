from pathlib import Path
import json
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_BAR_METRICS = [
    "test_accuracy",
    "test_loss",
    "global_beta_sq_mean",
    "global_low_frequency_ratio_mean",
    "global_total_energy_mean",
    "global_sigma_energy_ratio_mean",
    "global_grad_x_energy_ratio_mean",
    "global_grad_y_energy_ratio_mean",
    "global_grad_xy_energy_ratio_mean",
]

DEFAULT_PRETTY_NAMES = {
    "test_accuracy": "Test accuracy",
    "test_loss": "Test loss",
    "global_beta_sq_mean": "Beta squared",
    "global_low_frequency_ratio_mean": "Low-frequency ratio",
    "global_total_energy_mean": "Total energy",
    "global_sigma_energy_ratio_mean": "Sigma energy ratio",
    "global_grad_x_energy_ratio_mean": "Grad X energy ratio",
    "global_grad_y_energy_ratio_mean": "Grad Y energy ratio",
    "global_grad_xy_energy_ratio_mean": "Grad XY energy ratio",
}


def build_test_metrics_dataset(runs_dir: Path) -> pd.DataFrame:
    """
    Lit les métriques de test finales depuis summary.json pour tous les runs valides.
    """
    runs_dir = Path(runs_dir)
    rows = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"

        if not config_path.exists() or not summary_path.exists():
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        try:
            init_name = config["initializer"]["name"]
            seed = int(config["seed"])
            test_accuracy = float(summary["test_accuracy"])
            test_loss = float(summary["test_loss"])
        except KeyError as e:
            raise KeyError(f"Missing key while parsing {run_dir.name}: {e}")

        rows.append(
            {
                "init": init_name,
                "seed": seed,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
            }
        )

    if len(rows) == 0:
        raise ValueError(f"No valid test metrics found in {runs_dir}")

    df_test = pd.DataFrame(rows)
    df_test = df_test.drop_duplicates(subset=["init", "seed"]).reset_index(drop=True)

    return df_test


def build_final_dct_dataset(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Conserve, pour chaque couple (init, seed), la dernière ligne temporelle.
    """
    required_cols = [
        "init",
        "seed",
        "epoch",
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
        raise ValueError(f"Missing columns in df_all for final DCT dataset: {missing}")

    df_final_dct = (
        df_all[required_cols]
        .copy()
        .sort_values(["init", "seed", "epoch"])
        .groupby(["init", "seed"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    return df_final_dct


def build_final_barplot_dataset(df_all: pd.DataFrame, runs_dir: Path) -> pd.DataFrame:
    """
    Fusionne:
    - les métriques DCT finales (depuis df_all)
    - les métriques test finales (depuis summary.json)

    Sortie:
        1 ligne = 1 run final (init, seed)
    """
    df_final_dct = build_final_dct_dataset(df_all)
    df_test = build_test_metrics_dataset(runs_dir)

    df_final = pd.merge(
        df_final_dct,
        df_test,
        on=["init", "seed"],
        how="inner",
    )

    if df_final.empty:
        raise ValueError("Empty final barplot dataset after merging DCT and test metrics.")

    df_final = df_final.sort_values(["init", "seed"]).reset_index(drop=True)

    return df_final


def aggregate_barplot_metrics(
    df_final: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrège mean/std par initialisation.
    """
    metrics = metrics or DEFAULT_BAR_METRICS

    missing = [m for m in metrics if m not in df_final.columns]
    if missing:
        raise ValueError(f"Missing metrics in df_final: {missing}")

    df_mean = (
        df_final
        .groupby("init")[metrics]
        .mean()
        .reset_index()
    )

    df_std = (
        df_final
        .groupby("init")[metrics]
        .std()
        .reset_index()
        .fillna(0.0)
    )

    return df_mean, df_std


def plot_bar_metric(
    df_mean: pd.DataFrame,
    df_std: pd.DataFrame,
    metric: str,
    output_dir: Path,
    pretty_names: Optional[dict[str, str]] = None,
) -> Path:
    """
    Génère et sauvegarde un barplot pour une métrique finale.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metric not in df_mean.columns or metric not in df_std.columns:
        raise ValueError(f"Metric '{metric}' not found in aggregated barplot dataframes.")

    pretty_names = pretty_names or DEFAULT_PRETTY_NAMES
    title = pretty_names.get(metric, metric)

    x = np.arange(len(df_mean))
    y = df_mean[metric].values
    yerr = df_std[metric].values

    plt.figure(figsize=(8, 5))
    plt.bar(x, y, yerr=yerr, capsize=5)
    plt.xticks(x, df_mean["init"], rotation=30)
    plt.ylabel(title)
    plt.title(f"{title} (final)")
    plt.grid(axis="y")
    plt.tight_layout()

    output_path = output_dir / f"{metric}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_all_barplots(
    df_final: pd.DataFrame,
    output_dir: Path,
    metrics: Optional[list[str]] = None,
    pretty_names: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrège mean/std puis génère tous les barplots demandés.
    Retourne les dataframes agrégés pour inspection dans le notebook.
    """
    metrics = metrics or DEFAULT_BAR_METRICS
    pretty_names = pretty_names or DEFAULT_PRETTY_NAMES

    df_mean, df_std = aggregate_barplot_metrics(df_final, metrics=metrics)

    for metric in metrics:
        plot_bar_metric(
            df_mean=df_mean,
            df_std=df_std,
            metric=metric,
            output_dir=output_dir,
            pretty_names=pretty_names,
        )

    return df_mean, df_std