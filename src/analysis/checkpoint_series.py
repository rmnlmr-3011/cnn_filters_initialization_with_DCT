# Code permettant de construire une série temporelle des métriques d'analyse de filtres pour tous les checkpoints d'un run.

from pathlib import Path
from typing import List, Dict, Optional
import json
import re

import pandas as pd
import matplotlib.pyplot as plt

from src.utils.logging import get_logger


def parse_checkpoint_label(label: str) -> Optional[int]:
    """
    Convertit un label du type:
    - epoch_000 -> 0
    - epoch_001 -> 1
    - epoch_005 -> 5
    - final -> None
    """
    if label == "final":
        return None

    match = re.fullmatch(r"epoch_(\d+)", label)
    if match is None:
        raise ValueError(f"Unsupported checkpoint label: {label}")

    return int(match.group(1))


def collect_checkpoint_summaries(analysis_dir: Path) -> List[Dict]:
    """
    Parcourt les sous-dossiers de analysis/ et lit les filter_analysis_summary.json.
    """
    analysis_dir = Path(analysis_dir)

    rows: List[Dict] = []

    for subdir in sorted(analysis_dir.iterdir()):
        if not subdir.is_dir():
            continue

        summary_path = subdir / "filter_analysis_summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        label = subdir.name
        epoch = parse_checkpoint_label(label)

        row = {
            "checkpoint_label": label,
            "epoch": epoch,
            "is_final": label == "final",
            **summary,
        }
        rows.append(row)

    if len(rows) == 0:
        raise FileNotFoundError(
            f"No filter_analysis_summary.json files found in {analysis_dir}"
        )

    return rows


def build_checkpoint_series_dataframe(analysis_dir: Path) -> pd.DataFrame:
    """
    Construit un DataFrame une ligne = un checkpoint.
    """
    rows = collect_checkpoint_summaries(analysis_dir)
    df = pd.DataFrame(rows)

    # Tri: epochs croissants, puis final à la fin
    df["_sort_epoch"] = df["epoch"].fillna(float("inf"))
    df = df.sort_values(by=["_sort_epoch", "checkpoint_label"]).drop(columns=["_sort_epoch"])
    df = df.reset_index(drop=True)

    return df


def _plot_metric(df_plot: pd.DataFrame, x_col: str, y_col: str, output_path: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df_plot[x_col], df_plot[y_col], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_checkpoint_series_outputs(df: pd.DataFrame, analysis_dir: Path) -> None:
    """
    Sauvegarde le CSV récapitulatif et les graphes minimaux.
    """
    analysis_dir = Path(analysis_dir)
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = analysis_dir / "checkpoint_series_summary.csv"
    df.to_csv(csv_path, index=False)

    # Pour les graphes, on garde seulement les checkpoints avec epoch numérique
    df_plot = df[df["epoch"].notna()].copy()

    if df_plot.empty:
        raise ValueError("No numeric epochs available for plotting.")

    df_plot["epoch"] = df_plot["epoch"].astype(int)

    _plot_metric(
        df_plot=df_plot,
        x_col="epoch",
        y_col="global_total_energy_mean",
        output_path=plots_dir / "total_energy_mean_vs_epoch.png",
        title="Global total energy mean vs epoch",
        ylabel="Global total energy mean",
    )

    _plot_metric(
        df_plot=df_plot,
        x_col="epoch",
        y_col="global_low_frequency_ratio_mean",
        output_path=plots_dir / "low_frequency_ratio_mean_vs_epoch.png",
        title="Global low frequency ratio mean vs epoch",
        ylabel="Global low frequency ratio mean",
    )

    _plot_metric(
        df_plot=df_plot,
        x_col="epoch",
        y_col="global_beta_sq_mean",
        output_path=plots_dir / "beta_sq_mean_vs_epoch.png",
        title="Global beta² mean vs epoch",
        ylabel="Global beta² mean",
    )

    _plot_metric(
        df_plot=df_plot,
        x_col="epoch",
        y_col="global_sigma_energy_ratio_mean",
        output_path=plots_dir / "sigma_energy_ratio_mean_vs_epoch.png",
        title="Global sigma energy ratio mean vs epoch",
        ylabel="Global sigma energy ratio mean",
    )

    _plot_metric(
        df_plot=df_plot,
        x_col="epoch",
        y_col="global_grad_x_energy_ratio_mean",
        output_path=plots_dir / "grad_x_energy_ratio_mean_vs_epoch.png",
        title="Global grad_x energy ratio mean vs epoch",
        ylabel="Global grad_x energy ratio mean",
    )

    _plot_metric(
        df_plot=df_plot,
        x_col="epoch",
        y_col="global_grad_y_energy_ratio_mean",
        output_path=plots_dir / "grad_y_energy_ratio_mean_vs_epoch.png",
        title="Global grad_y energy ratio mean vs epoch",
        ylabel="Global grad_y energy ratio mean",
    )

    _plot_metric(
        df_plot=df_plot,
        x_col="epoch",
        y_col="global_grad_xy_energy_ratio_mean",
        output_path=plots_dir / "grad_xy_energy_ratio_mean_vs_epoch.png",
        title="Global grad_xy energy ratio mean vs epoch",
        ylabel="Global grad_xy energy ratio mean",
    )

    df_layers = build_layer_checkpoint_dataframe(analysis_dir)

    _plot_layer_epoch_heatmap(
        df_layers=df_layers,
        value_col="beta_sq_mean",
        output_path=plots_dir / "beta_sq_mean_layer_epoch_heatmap.png",
        title="Layer beta_sq mean across epochs",
    )

    _plot_layer_epoch_heatmap(
        df_layers=df_layers,
        value_col="low_frequency_ratio_mean",
        output_path=plots_dir / "low_frequency_ratio_mean_layer_epoch_heatmap.png",
        title="Layer low frequency ratio mean across epochs",
    )

    _plot_layer_epoch_heatmap(
        df_layers=df_layers,
        value_col="sigma_energy_ratio_mean",
        output_path=plots_dir / "sigma_energy_ratio_mean_layer_epoch_heatmap.png",
        title="Layer sigma energy ratio mean across epochs",
    )

    _plot_layer_epoch_heatmap(
        df_layers=df_layers,
        value_col="grad_x_energy_ratio_mean",
        output_path=plots_dir / "grad_x_energy_ratio_mean_layer_epoch_heatmap.png",
        title="Layer grad_x energy ratio mean across epochs",
    )

    _plot_layer_epoch_heatmap(
        df_layers=df_layers,
        value_col="grad_y_energy_ratio_mean",
        output_path=plots_dir / "grad_y_energy_ratio_mean_layer_epoch_heatmap.png",
        title="Layer grad_y energy ratio mean across epochs",
    )

    _plot_layer_epoch_heatmap(
        df_layers=df_layers,
        value_col="grad_xy_energy_ratio_mean",
        output_path=plots_dir / "grad_xy_energy_ratio_mean_layer_epoch_heatmap.png",
        title="Layer grad_xy energy ratio mean across epochs",
    )


# Création des heatmaps

def build_checkpoint_series(run_dir: Path) -> pd.DataFrame:
    """
    Fonction principale de l'étape C.
    """
    run_dir = Path(run_dir)
    analysis_dir = run_dir / "analysis"

    logger = get_logger("checkpoint_series")

    logger.info("===================================")
    logger.info("Checkpoint series summary")
    logger.info("===================================")
    logger.info(f"run_dir: {run_dir}")
    logger.info(f"analysis_dir: {analysis_dir}")
    logger.info("===================================")

    df = build_checkpoint_series_dataframe(analysis_dir)
    save_checkpoint_series_outputs(df, analysis_dir)

    logger.info(f"checkpoint_series_summary.csv saved to: {analysis_dir / 'checkpoint_series_summary.csv'}")
    logger.info(f"plots saved to: {analysis_dir / 'plots'}")
    logger.info("Checkpoint series summary completed successfully.")

    return df

def build_layer_checkpoint_dataframe(analysis_dir: Path) -> pd.DataFrame:
    """
    Construit un DataFrame avec une ligne = une couche pour un checkpoint donné.
    Colonnes attendues:
      - checkpoint_label
      - epoch
      - layer_idx
      - layer_name
      - métriques couche
    """
    analysis_dir = Path(analysis_dir)
    rows = []

    for subdir in sorted(analysis_dir.iterdir()):
        if not subdir.is_dir():
            continue

        label = subdir.name

        # Ignorer tout ce qui n'est pas un checkpoint de type epoch_XXX
        if not re.fullmatch(r"epoch_\d+", label):
            continue

        epoch = parse_checkpoint_label(label)

        layer_csv = subdir / "layer_metrics.csv"
        if not layer_csv.exists():
            continue

        layer_df = pd.read_csv(layer_csv)
        layer_df["checkpoint_label"] = label
        layer_df["epoch"] = epoch
        rows.append(layer_df)

    if len(rows) == 0:
        raise FileNotFoundError(
            f"No layer_metrics.csv files found in numeric checkpoint subdirectories of {analysis_dir}"
        )

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(by=["epoch", "layer_idx"]).reset_index(drop=True)
    return df




def _plot_layer_epoch_heatmap(
    df_layers: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
) -> None:
    pivot_df = df_layers.pivot(
        index="layer_idx",
        columns="epoch",
        values=value_col,
    ).sort_index()

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_df.values, aspect="auto", origin="lower")
    plt.colorbar(label=value_col)
    plt.xticks(
        ticks=range(len(pivot_df.columns)),
        labels=pivot_df.columns,
    )
    plt.yticks(
        ticks=range(len(pivot_df.index)),
        labels=pivot_df.index,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Layer index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()