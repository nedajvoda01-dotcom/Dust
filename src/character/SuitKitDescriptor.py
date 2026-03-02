"""SuitKitDescriptor — Stage 43 compact serialisable character appearance.

A ``SuitKitDescriptor`` is the *only* thing the server replicates to all
clients to describe a player's visual appearance.  All clients assemble
an identical-looking character from the same descriptor, so the server
never needs to send mesh or material data — only these small integers.

Wire format (per field, total 15 bytes)
-----------------------------------------
* ``helmet_id``  uint8  (0-255, N helmet variants)
* ``backpack_id``uint8
* ``chest_id``   uint8
* ``arm_id``     uint8
* ``leg_id``     uint8
* ``boot_id``    uint8
* ``accessory_id``uint8
* ``base_color`` uint8  (index into shared palette)
* ``accent_color``uint8
* ``roughness_var`` uint16 (0-65535 → 0.0-1.0)
* ``wear_amount``   uint16
* ``pattern_shift`` uint16

Derivation
----------
    playerSeed = hash(worldSeed, playerId)
    descriptor = SuitKitAssembler.build(playerSeed, module_counts)

Determinism: the same descriptor always produces the same visual on every
client, regardless of platform or run order.

Public helpers
--------------
SuitKitDescriptor.from_seed(player_seed, counts) — build from seed
SuitKitDescriptor.pack() → bytes
SuitKitDescriptor.unpack(data) → SuitKitDescriptor
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from src.core.DetRng import DetRng


# ---------------------------------------------------------------------------
# Module count defaults
# ---------------------------------------------------------------------------

_DEFAULT_MODULE_COUNTS = {
    "helmet":    6,
    "backpack":  5,
    "chest":     4,
    "arm":       4,
    "leg":       4,
    "boot":      3,
    "accessory": 8,
    "palette":  16,
}

# Wire format: 9 × uint8 + 3 × uint16  (9 + 6 = 15 bytes)
_PACK_FMT = ">BBBBBBBBBHHH"   # big-endian


@dataclass
class SuitKitDescriptor:
    """Compact appearance descriptor for one player."""

    helmet_id:     int = 0   # uint8
    backpack_id:   int = 0
    chest_id:      int = 0
    arm_id:        int = 0
    leg_id:        int = 0
    boot_id:       int = 0
    accessory_id:  int = 0
    base_color:    int = 0   # palette index uint8
    accent_color:  int = 0
    roughness_var: int = 0   # uint16
    wear_amount:   int = 0
    pattern_shift: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_seed(
        cls,
        player_seed:   int,
        module_counts: dict | None = None,
    ) -> "SuitKitDescriptor":
        """Build a deterministic descriptor from *player_seed*.

        Parameters
        ----------
        player_seed :
            Integer seed derived from ``hash(worldSeed, playerId)``.
        module_counts :
            Dict mapping module name → number of variants.  Falls back
            to built-in defaults.
        """
        mc  = dict(_DEFAULT_MODULE_COUNTS)
        if module_counts:
            mc.update(module_counts)

        rng = DetRng(player_seed)

        return cls(
            helmet_id     = rng.next_int(0, mc["helmet"]    - 1),
            backpack_id   = rng.next_int(0, mc["backpack"]  - 1),
            chest_id      = rng.next_int(0, mc["chest"]     - 1),
            arm_id        = rng.next_int(0, mc["arm"]       - 1),
            leg_id        = rng.next_int(0, mc["leg"]       - 1),
            boot_id       = rng.next_int(0, mc["boot"]      - 1),
            accessory_id  = rng.next_int(0, mc["accessory"] - 1),
            base_color    = rng.next_int(0, mc["palette"]   - 1),
            accent_color  = rng.next_int(0, mc["palette"]   - 1),
            roughness_var = rng.next_int(0, 65535),
            wear_amount   = rng.next_int(0, 65535),
            pattern_shift = rng.next_int(0, 65535),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def pack(self) -> bytes:
        """Serialise to 15-byte big-endian wire format."""
        return struct.pack(
            _PACK_FMT,
            self.helmet_id    & 0xFF,
            self.backpack_id  & 0xFF,
            self.chest_id     & 0xFF,
            self.arm_id       & 0xFF,
            self.leg_id       & 0xFF,
            self.boot_id      & 0xFF,
            self.accessory_id & 0xFF,
            self.base_color   & 0xFF,
            self.accent_color & 0xFF,
            self.roughness_var & 0xFFFF,
            self.wear_amount   & 0xFFFF,
            self.pattern_shift & 0xFFFF,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "SuitKitDescriptor":
        """Deserialise from 15-byte wire format."""
        fields = struct.unpack(_PACK_FMT, data)
        return cls(
            helmet_id     = fields[0],
            backpack_id   = fields[1],
            chest_id      = fields[2],
            arm_id        = fields[3],
            leg_id        = fields[4],
            boot_id       = fields[5],
            accessory_id  = fields[6],
            base_color    = fields[7],
            accent_color  = fields[8],
            roughness_var = fields[9],
            wear_amount   = fields[10],
            pattern_shift = fields[11],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SuitKitDescriptor):
            return NotImplemented
        return self.pack() == other.pack()
