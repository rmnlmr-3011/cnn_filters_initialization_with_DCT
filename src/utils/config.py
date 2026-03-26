# Code permettant de charger un fichier de config Yaml

from pathlib import Path
from typing import Union
import yaml


def load_config(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)