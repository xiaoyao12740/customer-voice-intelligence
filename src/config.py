from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    config = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))
    for section, key in (("data", "raw_dir"), ("data", "processed_dir"), ("outputs", "root"), ("outputs", "model_path")):
        config[section][key] = str(PROJECT_ROOT / config[section][key])
    return config


def database_url() -> str:
    return os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'voc.db'}")

