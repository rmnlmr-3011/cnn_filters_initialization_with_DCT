# Ce fichier contient la logique d'analyse d'un modèle Keras

from pathlib import Path
import pandas as pd

from src.analysis.model_filters import get_conv3x3_layers, iter_conv_kernels
from src.analysis.filter_metrics import (
    summarize_kernel_metrics,
    summarize_filter_metrics,
    summarize_layer_metrics,
)


def analyze_model(model):
    """
    Analyse un modèle Keras :
    - métriques par noyau
    - métriques par filtre (out_channel)
    - métriques par couche
    - résumé global
    """

    # =========================
    # 1. Kernel-level
    # =========================

    kernel_metrics = []

    for kernel_info in iter_conv_kernels(model):
        km = summarize_kernel_metrics(
            layer_idx=kernel_info["layer_idx"],
            layer_name=kernel_info["layer_name"],
            in_channel=kernel_info["in_channel"],
            out_channel=kernel_info["out_channel"],
            kernel_2d=kernel_info["kernel"],
        )
        kernel_metrics.append(km)

    if len(kernel_metrics) == 0:
        raise ValueError("No 3x3 kernels found in model.")

    kernel_df = pd.DataFrame(kernel_metrics)

    # =========================
    # 2. Filter-level (pandas)
    # =========================

    def _aggregate_filter(group_df):
        fm = summarize_filter_metrics(group_df.to_dict("records"))
        return pd.Series(fm)

    filter_df = (
        kernel_df
        .groupby(["layer_idx", "layer_name", "out_channel"], as_index=False)
        .apply(_aggregate_filter)
        .reset_index(drop=True)
    )

    # =========================
    # 3. Layer-level (pandas)
    # =========================

    def _aggregate_layer(group_df):
        lm = summarize_layer_metrics(group_df.to_dict("records"))
        return pd.Series(lm)

    layer_df = (
        kernel_df
        .groupby(["layer_idx", "layer_name"], as_index=False)
        .apply(_aggregate_layer)
        .reset_index(drop=True)
    )

    # =========================
    # 4. Summary global
    # =========================

    summary = {
        "n_conv3x3_layers": int(layer_df.shape[0]),
        "n_total_kernels": int(kernel_df.shape[0]),
        "n_total_filters": int(filter_df.shape[0]),
        "global_total_energy_mean": float(kernel_df["total_energy"].mean()),
        "global_total_energy_std": float(kernel_df["total_energy"].std()),
        "global_low_frequency_ratio_mean": float(kernel_df["low_frequency_ratio"].mean()),
        "global_low_frequency_ratio_std": float(kernel_df["low_frequency_ratio"].std()),
        "global_beta_sq_mean": float(kernel_df["beta_sq"].mean()),
        "global_beta_sq_std": float(kernel_df["beta_sq"].std()),
    }

    return kernel_df, filter_df, layer_df, summary