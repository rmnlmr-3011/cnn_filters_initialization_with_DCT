# Code pour sauvegarder les résultats (config d'un run (JSON), historique d'entrainement (CSV), filtres/poids (NPZ))

import json
from pathlib import Path
import pandas as pd
from datetime import datetime


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_history_csv(history, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(path, index=False)


def make_run_dir(base_dir: str = "runs", prefix: str = "run", timestamp: bool = True) -> Path:
    if timestamp:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{prefix}_{suffix}"
    else:
        run_name = prefix

    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir