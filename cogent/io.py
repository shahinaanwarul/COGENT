from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple
import json

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .semantic import load_embedding_map


def save_preprocessor_state(path: str | Path, preprocessor) -> None:
    state = {
        "scaling": None if preprocessor.scaling is None else {
            "columns": list(preprocessor.scaling.columns),
            "means": preprocessor.scaling.means,
            "scales": preprocessor.scaling.scales,
        },
        "country_vocab": preprocessor.country_vocab,
        "platform_vocab": preprocessor.platform_vocab,
        "label_thresholds": preprocessor.label_thresholds,
    }
    Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_preprocessor_state(path: str | Path) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_prepared_directory(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    p = Path(path)
    train = pd.read_csv(p / "train.csv", parse_dates=["Date"])
    val = pd.read_csv(p / "val.csv", parse_dates=["Date"])
    test = pd.read_csv(p / "test.csv", parse_dates=["Date"])
    state = load_preprocessor_state(p / "preprocessor_state.json")
    return train, val, test, state
