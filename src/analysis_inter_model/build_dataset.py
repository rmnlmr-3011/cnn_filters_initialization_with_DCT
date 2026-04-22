from pathlib import Path
import pandas as pd
import json
import re


def parse_run(run_dir: Path) -> pd.DataFrame:
    run_dir = Path(run_dir)

    # =========================
    # 1) Metadata depuis config.json
    # =========================
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    try:
        init_name = config["initializer"]["name"]
    except KeyError as e:
        raise KeyError(f"Could not read initializer name from {config_path}: {e}")

    try:
        seed = int(config["seed"])
    except KeyError as e:
        raise KeyError(f"Could not read seed from {config_path}: {e}")

    # =========================
    # 2) History
    # =========================
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.csv in {run_dir}")

    df_hist = pd.read_csv(history_path)

    # Keras history.csv n'a pas toujours la colonne epoch
    if "epoch" not in df_hist.columns:
        # epochs d'entraînement = 1..N
        df_hist["epoch"] = range(1, len(df_hist) + 1)

    # On ne garde que ce qui nous sert pour A
    hist_cols = ["epoch", "val_loss", "val_accuracy"]
    missing_hist_cols = [c for c in hist_cols if c not in df_hist.columns]
    if missing_hist_cols:
        raise ValueError(
            f"Missing columns in history.csv for {run_dir.name}: {missing_hist_cols}"
        )

    df_hist = df_hist[hist_cols].copy()
    df_hist = df_hist.drop_duplicates(subset=["epoch"])
    df_hist = df_hist.sort_values("epoch").reset_index(drop=True)

    # =========================
    # 3) Checkpoint series summary
    # =========================
    dct_path = run_dir / "analysis" / "checkpoint_series_summary.csv"
    if not dct_path.exists():
        raise FileNotFoundError(f"Missing checkpoint_series_summary.csv in {run_dir}")

    df_dct = pd.read_csv(dct_path)

    # On garde uniquement les checkpoints numériques (epochs réels)
    if "epoch" not in df_dct.columns:
        raise ValueError(f"Missing 'epoch' column in {dct_path}")

    dct_cols = [
        "epoch",
        "global_beta_sq_mean",
        "global_low_frequency_ratio_mean",
        "global_total_energy_mean",
        "global_sigma_energy_ratio_mean",
        "global_grad_x_energy_ratio_mean",
        "global_grad_y_energy_ratio_mean",
        "global_grad_xy_energy_ratio_mean",
    ]
    missing_dct_cols = [c for c in dct_cols if c not in df_dct.columns]
    if missing_dct_cols:
        raise ValueError(
            f"Missing columns in checkpoint_series_summary.csv for {run_dir.name}: "
            f"{missing_dct_cols}"
        )

    df_dct = df_dct[dct_cols].copy()
    df_dct = df_dct[df_dct["epoch"].notna()].copy()
    df_dct["epoch"] = df_dct["epoch"].astype(int)
    df_dct = df_dct.drop_duplicates(subset=["epoch"])
    df_dct = df_dct.sort_values("epoch").reset_index(drop=True)

    # =========================
    # 4) Merge propre sur epoch
    # =========================
    df = pd.merge(df_hist, df_dct, on="epoch", how="inner")

    if df.empty:
        raise ValueError(
            f"Empty merge for {run_dir.name}. "
            f"history epochs={df_hist['epoch'].tolist()} "
            f"dct epochs={df_dct['epoch'].tolist()}"
        )

    # =========================
    # 5) Ajout metadata
    # =========================
    df["init"] = init_name
    df["seed"] = seed

    # Ordre final des colonnes
    df = df[
        [
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
    ].copy()

    return df


# Fonction qui permet de construire un dataset à partir d'un dossier de runs, en utilisant la fonction parse_run pour chaque run
def build_dataset(runs_dir: Path) -> pd.DataFrame:
    runs_dir = Path(runs_dir)

    dfs = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        try:
            df_run = parse_run(run_dir)
            dfs.append(df_run)
            print(
                f"[OK] {run_dir.name} -> init={df_run['init'].iloc[0]}, "
                f"seed={df_run['seed'].iloc[0]}, n_rows={len(df_run)}"
            )
        except Exception as e:
            print(f"[WARNING] skipped {run_dir.name}: {e}")

    if len(dfs) == 0:
        raise ValueError(f"No valid runs found in {runs_dir}")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values(by=["init", "seed", "epoch"]).reset_index(drop=True)

    return df_all

