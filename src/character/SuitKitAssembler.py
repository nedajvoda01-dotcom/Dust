"""SuitKitAssembler — Stage 43 procedural character assembly.

Converts a :class:`~src.character.SuitKitDescriptor.SuitKitDescriptor` into
a fully assembled :class:`SuitKit` — the runtime representation of a
player's visual/physical suit.

Assembly is *cheap*: it reads small integer IDs from the descriptor, looks up
module properties from the per-module tables, and returns a ``SuitKit``.  No
heavy computation, no textures, no meshes — only parameter structs.

Procedural material parameters
-------------------------------
* ``roughness`` is derived from ``descriptor.roughness_var / 65535`` scaled
  into [0.2, 0.9] — a plausible range for a worn suit surface.
* ``wear_amount`` → edge-wear intensity [0..1].
* ``pattern_shift`` → UV / tri-planar pattern phase [0..1].

Physical micro-variations
--------------------------
All variations are within the bounds specified in §6.5:
* Suit mass ±5–8% of base; applied as an additive offset to the archetype
  total mass.  Stored in ``SuitKit.extra_mass_kg``.
* COM shift ±2–4% back/up along the body.  Stored in
  ``SuitKit.com_shift_norm`` as a (back, up) tuple in normalised units.
* Inertia scale ±5%.  Stored in ``SuitKit.inertia_scale``.

Public API
----------
SuitKitAssembler(config=None)
  .assemble(descriptor, archetype) → SuitKit
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from src.character.SuitKitDescriptor import SuitKitDescriptor
from src.character.BodyArchetypeDB   import BodyArchetype


# ---------------------------------------------------------------------------
# Runtime SuitKit
# ---------------------------------------------------------------------------

@dataclass
class SuitModule:
    """Runtime representation of one assembled suit module."""
    module_type: str   # "helmet", "backpack", …
    variant_id:  int
    # Acoustic properties (used by SuitAcousticBinder)
    friction_noise_weight: float = 0.0   # cloth/strap contribution
    modal_freq_hz:         float = 0.0   # metal/plastic resonance
    impact_lf_weight:      float = 0.0   # low-freq thump contribution


@dataclass
class SuitMaterialParams:
    """Procedural material parameters (no textures; pure scalars)."""
    roughness:     float = 0.5
    wear_amount:   float = 0.2
    pattern_shift: float = 0.0
    base_color_id: int   = 0
    accent_color_id: int = 0


@dataclass
class SuitKit:
    """Fully assembled character suit for one player."""
    modules:           list           # List[SuitModule]
    material:          SuitMaterialParams = None
    # Physical micro-variation (within §6.5 bounds)
    extra_mass_kg:     float = 0.0    # additive to archetype total mass
    com_shift_norm:    Tuple[float, float] = (0.0, 0.0)  # (back, up) [0..1]
    inertia_scale:     float = 1.0    # ± 5%

    def __post_init__(self):
        if self.material is None:
            self.material = SuitMaterialParams()


# ---------------------------------------------------------------------------
# Module property tables
# ---------------------------------------------------------------------------

# Each entry: (friction_noise_weight, modal_freq_hz, impact_lf_weight)
_HELMET_TABLE = [
    (0.0, 1200.0, 0.1),
    (0.0, 1400.0, 0.1),
    (0.0,  900.0, 0.15),
    (0.1,  800.0, 0.2),
    (0.0, 1600.0, 0.05),
    (0.0, 1100.0, 0.1),
]
_BACKPACK_TABLE = [
    (0.3,  200.0, 0.8),
    (0.2,  250.0, 0.7),
    (0.4,  180.0, 0.9),
    (0.1,  300.0, 0.6),
    (0.3,  220.0, 0.75),
]
_CHEST_TABLE = [
    (0.5, 600.0, 0.3),
    (0.4, 700.0, 0.2),
    (0.6, 550.0, 0.35),
    (0.5, 650.0, 0.3),
]
_ARM_TABLE = [
    (0.6, 800.0, 0.1),
    (0.5, 750.0, 0.15),
    (0.7, 900.0, 0.1),
    (0.5, 850.0, 0.1),
]
_LEG_TABLE = [
    (0.6, 700.0, 0.2),
    (0.5, 650.0, 0.25),
    (0.7, 750.0, 0.2),
    (0.6, 680.0, 0.2),
]
_BOOT_TABLE = [
    (0.7, 400.0, 0.5),
    (0.6, 350.0, 0.6),
    (0.8, 450.0, 0.45),
]
_ACCESSORY_TABLE = [
    (0.2, 1800.0, 0.0),
    (0.1, 2000.0, 0.0),
    (0.3, 1600.0, 0.0),
    (0.1, 2200.0, 0.0),
    (0.2, 1900.0, 0.0),
    (0.1, 2100.0, 0.0),
    (0.3, 1700.0, 0.0),
    (0.2, 1950.0, 0.0),
]

# Physical extra-mass per backpack variant (kg)
_BACKPACK_MASS_KG = [4.0, 3.5, 5.0, 3.0, 4.5]
# COM shift per backpack variant: (back, up) normalised
_BACKPACK_COM = [(0.02, 0.01), (0.015, 0.01), (0.025, 0.015),
                 (0.010, 0.005), (0.020, 0.012)]


def _lookup(table: list, idx: int) -> tuple:
    return table[idx % len(table)]


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class SuitKitAssembler:
    """Assembles a :class:`SuitKit` from a :class:`SuitKitDescriptor`.

    Parameters
    ----------
    config :
        Optional dict; reads ``suit.*`` sub-keys.
    """

    _DEFAULT_MASS_VAR_PCT  = 0.06   # ±6%
    _DEFAULT_COM_SHIFT_MAX = 0.03   # normalised units
    _DEFAULT_INERTIA_VAR   = 0.05   # ±5%

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        scfg = cfg.get("suit", {}) or {}
        self._mass_var_pct  = float(scfg.get("mass_variation_pct", self._DEFAULT_MASS_VAR_PCT))
        self._com_max       = float(scfg.get("com_shift_max",      self._DEFAULT_COM_SHIFT_MAX))
        self._inertia_var   = float(scfg.get("inertia_var",        self._DEFAULT_INERTIA_VAR))

    def assemble(
        self,
        descriptor: SuitKitDescriptor,
        archetype:  Optional[BodyArchetype] = None,
    ) -> SuitKit:
        """Build a SuitKit from *descriptor*.

        Parameters
        ----------
        descriptor :
            Compact appearance descriptor (from the network or from seed).
        archetype :
            BodyArchetype for mass baseline; uses defaults if None.
        """
        # --- Build modules ---
        modules = [
            self._make_module("helmet",    descriptor.helmet_id,    _HELMET_TABLE),
            self._make_module("backpack",  descriptor.backpack_id,  _BACKPACK_TABLE),
            self._make_module("chest",     descriptor.chest_id,     _CHEST_TABLE),
            self._make_module("arm",       descriptor.arm_id,       _ARM_TABLE),
            self._make_module("leg",       descriptor.leg_id,       _LEG_TABLE),
            self._make_module("boot",      descriptor.boot_id,      _BOOT_TABLE),
            self._make_module("accessory", descriptor.accessory_id, _ACCESSORY_TABLE),
        ]

        # --- Material params ---
        roughness     = 0.2 + (descriptor.roughness_var / 65535.0) * 0.7
        wear          = descriptor.wear_amount   / 65535.0
        pattern_shift = descriptor.pattern_shift / 65535.0
        material = SuitMaterialParams(
            roughness      = roughness,
            wear_amount    = wear,
            pattern_shift  = pattern_shift,
            base_color_id  = descriptor.base_color,
            accent_color_id= descriptor.accent_color,
        )

        # --- Physical micro-variations ---
        bp_idx       = descriptor.backpack_id % len(_BACKPACK_MASS_KG)
        base_suit_mass = _BACKPACK_MASS_KG[bp_idx]

        # Roughness var as a tiny mass variation proxy (deterministic)
        mass_frac    = (descriptor.roughness_var / 65535.0 - 0.5) * 2.0  # [-1, 1]
        extra_mass   = base_suit_mass * (1.0 + mass_frac * self._mass_var_pct)

        bp_com       = _BACKPACK_COM[bp_idx]
        com_scale    = descriptor.wear_amount / 65535.0  # [0, 1]
        com_shift    = (
            bp_com[0] * (1.0 + (com_scale - 0.5) * 0.5),
            bp_com[1] * (1.0 + (com_scale - 0.5) * 0.3),
        )

        inertia_frac = (descriptor.pattern_shift / 65535.0 - 0.5) * 2.0
        inertia_scale = 1.0 + inertia_frac * self._inertia_var

        return SuitKit(
            modules       = modules,
            material      = material,
            extra_mass_kg = extra_mass,
            com_shift_norm= com_shift,
            inertia_scale = inertia_scale,
        )

    @staticmethod
    def _make_module(module_type: str, idx: int, table: list) -> SuitModule:
        props = _lookup(table, idx)
        return SuitModule(
            module_type           = module_type,
            variant_id            = idx,
            friction_noise_weight = props[0],
            modal_freq_hz         = props[1],
            impact_lf_weight      = props[2],
        )
