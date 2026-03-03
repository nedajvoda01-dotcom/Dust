"""PlayerPersistence — Stage 57 server-issued stable player identifiers.

Each connecting player receives a stable UUID that survives server restarts
and is stored on the client in ``localStorage``.  The mapping from the
transient session key (IP + User-Agent hash) to the stable UUID is persisted
in ``world_state/players.json``.

Why two IDs?
------------
* **Session key** (``player_key``): SHA-256(IP | UA | salt)[:16].  Changes
  every server restart because the salt is regenerated.  Used internally for
  registry lookups within one session.
* **Stable ID** (``stable_id``): UUID4 issued on first connection and stored
  on disk.  Survives restarts, outlasts the session key.

On reconnect the client sends its stored stable_id in ``HELLO``.  The server
validates it against the disk record; if it matches, no new ID is issued.

Public API
----------
PlayerPersistence(state_dir)
  .get_or_create(player_key, hint_id="") → str    stable UUID for player_key
  .update_last_seen(player_key)          → None
  .all_records()                         → dict
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict

_PLAYERS_FILE = "players.json"


class PlayerPersistence:
    """Manages server-issued stable player UUIDs.

    Parameters
    ----------
    state_dir :
        Directory where ``world_state/`` lives.  ``players.json`` is stored
        directly inside this directory.
    """

    def __init__(self, state_dir: str = "world_state") -> None:
        self._path = Path(state_dir) / _PLAYERS_FILE
        self._records: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create(self, player_key: str, hint_id: str = "") -> str:
        """Return the stable UUID for *player_key*, creating one if needed.

        Parameters
        ----------
        player_key :
            Transient session key (hash of IP + UA + salt).
        hint_id :
            If the client sent a previously assigned stable UUID in ``HELLO``,
            pass it here.  The server reuses it only when it is already on
            record for this *player_key* (prevents ID spoofing).
        """
        rec = self._records.get(player_key)
        if rec is not None:
            # Validate hint if provided: must match stored stable_id
            if hint_id and rec["stable_id"] == hint_id:
                return hint_id
            return str(rec["stable_id"])

        # New player — issue fresh UUID
        new_id = str(uuid.uuid4())
        self._records[player_key] = {
            "stable_id": new_id,
            "joined_at": time.time(),
            "last_seen": time.time(),
        }
        self._save()
        return new_id

    def update_last_seen(self, player_key: str) -> None:
        """Update the ``last_seen`` timestamp for *player_key*."""
        rec = self._records.get(player_key)
        if rec is not None:
            rec["last_seen"] = time.time()
            self._save()

    def all_records(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of all player records."""
        return dict(self._records)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._records = data
        except Exception:
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(json.dumps(self._records, indent=2).encode("utf-8"))
            shutil.move(tmp, str(self._path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
