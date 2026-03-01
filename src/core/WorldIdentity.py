"""WorldIdentity — persistent world seed and unique world ID.

Saves and loads a small JSON record that uniquely identifies a world
instance and ties it to a deterministic generation seed.

Schema (world_identity.json):
  {
    "schemaVersion": 1,
    "worldId": "<uuid4>",
    "seed": <int>,
    "createdAt": "<ISO-8601 UTC>"
  }

If the file is missing or its schemaVersion does not match the current
version a fresh identity is generated and saved.

Public API
----------
WorldIdentity.load_or_create(storage, default_seed, schema_version)
  .world_id   → str    unique UUID for this world
  .seed       → int    world generation seed
  .created_at → str    ISO-8601 creation timestamp
  .save(storage)       persist / overwrite the identity file
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from src.core.Logger import Logger
from src.core.PersistentStorage import PersistentStorage

_TAG = "WorldIdentity"
_FILENAME = "world_identity.json"
_CURRENT_SCHEMA = 1


class WorldIdentity:
    """Persistent world identity: seed + UUID + creation timestamp."""

    def __init__(self) -> None:
        self.world_id:       str = ""
        self.seed:           int = 42
        self.created_at:     str = ""
        self.schema_version: int = _CURRENT_SCHEMA

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load_or_create(
        cls,
        storage: PersistentStorage,
        default_seed: int = 42,
        schema_version: int = _CURRENT_SCHEMA,
    ) -> "WorldIdentity":
        """Load from *storage*, or create a new identity if none exists.

        Parameters
        ----------
        storage:
            Initialised PersistentStorage instance.
        default_seed:
            Seed used when generating a new identity.
        schema_version:
            Expected schema version; mismatches cause a new identity.
        """
        identity = cls()
        data = storage.read_json(_FILENAME)

        if data is not None:
            stored_ver = data.get("schemaVersion", 0)
            if stored_ver == schema_version:
                identity.world_id       = str(data.get("worldId", ""))
                identity.seed           = int(data.get("seed", default_seed))
                identity.created_at     = str(data.get("createdAt", ""))
                identity.schema_version = stored_ver
                Logger.info(
                    _TAG,
                    f"Loaded world: id={identity.world_id} seed={identity.seed}",
                )
                return identity
            Logger.warn(
                _TAG,
                f"Schema mismatch (stored={stored_ver} expected={schema_version}); "
                "generating new world identity",
            )

        # Generate fresh identity
        identity.world_id       = str(uuid.uuid4())
        identity.seed           = default_seed
        identity.created_at     = datetime.now(timezone.utc).isoformat()
        identity.schema_version = schema_version
        identity.save(storage)
        Logger.info(_TAG, f"Created world: id={identity.world_id} seed={identity.seed}")
        return identity

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, storage: PersistentStorage) -> None:
        """Write the identity to *storage* atomically."""
        storage.write_json_atomic(
            _FILENAME,
            {
                "schemaVersion": self.schema_version,
                "worldId":       self.world_id,
                "seed":          self.seed,
                "createdAt":     self.created_at,
            },
        )
