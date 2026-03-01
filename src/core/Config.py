"""Config — loads CONFIG_DEFAULTS.json; warns on missing keys."""
from __future__ import annotations

import json
import os
from typing import Any

from src.core.Logger import Logger

_TAG = "Config"


class Config:
    def __init__(self, path: str | None = None) -> None:
        if path is None:
            root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            path = os.path.join(root, "config", "CONFIG_DEFAULTS.json")
        self._path = path
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                self._data = json.load(fh)
            Logger.info(_TAG, f"Loaded config: {self._path}")
        except FileNotFoundError:
            Logger.warn(_TAG, f"Config file not found: {self._path}. Using defaults.")
            self._data = {}
        except json.JSONDecodeError as exc:
            Logger.error(_TAG, f"Config parse error: {exc}")
            self._data = {}

    def get(self, *keys: str, default: Any = None) -> Any:
        """Navigate nested dict by keys; warns if path not found."""
        node = self._data
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                Logger.warn(_TAG, f"Missing config key: {' > '.join(keys)} (using default={default!r})")
                return default
            node = node[k]
        return node
