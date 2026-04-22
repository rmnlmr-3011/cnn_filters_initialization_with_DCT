# src/analysis/plot_correlations.py

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import json


def build_final_scatter_dataset(df_all: pd.DataFrame, runs_root: Path) -> pd.DataFrame:
    runs_root = Path(runs_root)

    # 1) Dernières métriques DCT par run
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
        raise ValueError(f"Missing columns in df_all for scatter dataset: {missing}")

    df_final_dct = (
        df_all[required_cols]
        .copy()
        .sort_values(["init", "seed", "epoch"])
        .groupby(["init", "seed"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    # 2) Métriques test depuis summary.json
    rows = []

    for run_dir in sorted(runs_root.iterdir()):
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

        rows.append({
            "init": config["initializer"]["name"],
            "seed": int(config["seed"]),
            "test_accuracy": float(summary["test_accuracy"]),
            "test_loss": float(summary["test_loss"]),
        })

    if len(rows) == 0:
        raise ValueError(f"No valid summary.json rows found in {runs_root}")

    df_test = pd.DataFrame(rows).drop_duplicates(subset=["init", "seed"]).reset_index(drop=True)

    # 3) Fusion
    df_scatter = pd.merge(
        df_final_dct,
        df_test,
        on=["init", "seed"],
        how="inner",
    )

    if df_scatter.empty:
        raise ValueError("Empty scatter dataset after merging final DCT and test metrics.")

    return df_scatter


PRETTY_NAMES = {
    "global_low_frequency_ratio_mean": "Low-frequency ratio",
    "global_beta_sq_mean": "Beta squared",
    "global_sigma_energy_ratio_mean": "Sigma energy ratio",
    "global_grad_x_energy_ratio_mean": "Grad X energy ratio",
    "global_grad_y_energy_ratio_mean": "Grad Y energy ratio",
    "global_grad_xy_energy_ratio_mean": "Grad XY energy ratio",
    "test_accuracy": "Test accuracy",
    "test_loss": "Test loss",
}


def compute_pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return np.nan

    return np.corrcoef(x, y)[0, 1]


def plot_correlation(
    df,
    x_col,
    y_col="test_accuracy",
    title=None,
    save_path=None,
):
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in dataframe.")
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in dataframe.")

    plt.figure(figsize=(6, 5))

    inits = sorted(df["init"].unique())

    for init in inits:
        subset = df[df["init"] == init]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            label=init,
            alpha=0.8,
        )

    x = df[x_col].values
    y = df[y_col].values

    if len(x) >= 2:
        coeffs = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        plt.plot(x_line, y_line, linestyle="--")

    r = compute_pearson_r(x, y)

    label_x = PRETTY_NAMES.get(x_col, x_col)
    label_y = PRETTY_NAMES.get(y_col, y_col)

    plt.xlabel(label_x)
    plt.ylabel(label_y)

    if title is None:
        title = f"{label_x} vs {label_y} (r={r:.3f})"

    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()


def plot_all_correlations(df, save_dir=None):
    metrics = [
        "global_low_frequency_ratio_mean",
        "global_beta_sq_mean",
        "global_sigma_energy_ratio_mean",
        "global_grad_x_energy_ratio_mean",
        "global_grad_y_energy_ratio_mean",
        "global_grad_xy_energy_ratio_mean",
    ]

    for metric in metrics:
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f"corr_{metric}.png"

        plot_correlation(
            df,
            x_col=metric,
            y_col="test_accuracy",
            save_path=save_path,
        )