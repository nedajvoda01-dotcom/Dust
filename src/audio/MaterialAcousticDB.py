"""MaterialAcousticDB — Stage 36 per-material acoustic profiles.

Each material has a :class:`MaterialAcousticProfile` that describes its
resonant behaviour (modal frequencies, weights, decay times) and its
surface character (roughness, graininess, damping).

The :class:`MaterialAcousticDB` look-up table maps integer material IDs to
profiles.  Material IDs mirror those used in ``ProceduralAudioSystem`` and
``ProceduralMaterialSystem``.

Material IDs
------------
0  MAT_DUST         — soft granular layer
1  MAT_BASALT       — dense volcanic rock
2  MAT_DEBRIS       — loose rubble / scree
3  MAT_FRACT        — fractured rock
4  MAT_ICE          — ice film / compacted snow
5  MAT_SNOW         — loose snowpack
6  MAT_PLATE        — large rock plate (bulk resonator use)

Public API
----------
MaterialAcousticDB()
  .get(mat_id: int) -> MaterialAcousticProfile
  .mix(profile_a, profile_b, ratio) -> MaterialAcousticProfile
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Material ID constants (mirror ProceduralMaterialSystem / ProceduralAudioSystem)
# ---------------------------------------------------------------------------

MAT_DUST   = 0
MAT_BASALT = 1
MAT_DEBRIS = 2
MAT_FRACT  = 3
MAT_ICE    = 4
MAT_SNOW   = 5
MAT_PLATE  = 6


# ---------------------------------------------------------------------------
# MaterialAcousticProfile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialAcousticProfile:
    """Acoustic characterisation of one surface material.

    Attributes
    ----------
    density :
        Normalised material density [0..1]; heavier materials resonate lower.
    stiffness :
        Normalised stiffness [0..1]; stiffer → higher modal frequencies.
    damping :
        Internal damping coefficient [0..1]; high = fast amplitude decay.
    roughness :
        Surface roughness [0..1]; drives friction-noise spectral content.
    graininess :
        Granular texture [0..1]; contributes to crumble / sifting bursts.
    modal_frequencies : tuple of float
        Centre frequencies [Hz] of the resonant modes (normalised to 1 kHz).
        Stored as normalised ratios; actual pitch depends on stiffness * density.
    modal_weights : tuple of float
        Relative excitation weight of each mode (must sum ≤ 1.0 across modes).
    modal_decay : tuple of float
        Decay time constants [s] for each mode's amplitude envelope.
    noise_color :
        Preferred noise colour for friction excitation:
        "white", "pink", or "brown".
    """
    density:           float
    stiffness:         float
    damping:           float
    roughness:         float
    graininess:        float
    modal_frequencies: Tuple[float, ...]
    modal_weights:     Tuple[float, ...]
    modal_decay:       Tuple[float, ...]
    noise_color:       str = "pink"

    def __post_init__(self) -> None:
        assert len(self.modal_frequencies) == len(self.modal_weights) == len(self.modal_decay), (
            f"modal_frequencies (len={len(self.modal_frequencies)}), "
            f"modal_weights (len={len(self.modal_weights)}) and "
            f"modal_decay (len={len(self.modal_decay)}) must have equal length"
        )


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_PROFILES: Dict[int, MaterialAcousticProfile] = {
    MAT_DUST: MaterialAcousticProfile(
        density=0.2,
        stiffness=0.1,
        damping=0.95,
        roughness=0.6,
        graininess=0.8,
        # Dust: broad noise, weak low modes, very fast decay
        modal_frequencies=(80.0,  160.0, 320.0),
        modal_weights=    (0.10,  0.06,  0.04),
        modal_decay=      (0.015, 0.010, 0.007),
        noise_color="pink",
    ),

    MAT_BASALT: MaterialAcousticProfile(
        density=0.85,
        stiffness=0.90,
        damping=0.25,
        roughness=0.45,
        graininess=0.15,
        # Basalt: clear low modes, metallic overtone on hard impact, long decay
        modal_frequencies=(120.0, 310.0, 580.0, 940.0, 1400.0,
                           1900.0, 2600.0, 3500.0),
        modal_weights=    (0.30,  0.22,  0.18,  0.12,  0.08,
                           0.05,  0.03,  0.02),
        modal_decay=      (0.45,  0.30,  0.20,  0.14,  0.09,
                           0.06,  0.04,  0.03),
        noise_color="pink",
    ),

    MAT_DEBRIS: MaterialAcousticProfile(
        density=0.50,
        stiffness=0.40,
        damping=0.60,
        roughness=0.75,
        graininess=0.90,
        # Loose debris: medium modes, lots of grain cascade
        modal_frequencies=(100.0, 240.0, 480.0, 800.0),
        modal_weights=    (0.20,  0.18,  0.14,  0.08),
        modal_decay=      (0.08,  0.06,  0.04,  0.03),
        noise_color="pink",
    ),

    MAT_FRACT: MaterialAcousticProfile(
        density=0.75,
        stiffness=0.70,
        damping=0.40,
        roughness=0.65,
        graininess=0.55,
        # Fractured rock: medium-low modes + high-freq micro-clicks
        modal_frequencies=(150.0, 380.0, 700.0, 1100.0, 1800.0,
                           2800.0),
        modal_weights=    (0.25,  0.20,  0.16,  0.12,   0.08,
                           0.05),
        modal_decay=      (0.20,  0.14,  0.09,  0.06,   0.04,
                           0.025),
        noise_color="white",
    ),

    MAT_ICE: MaterialAcousticProfile(
        density=0.45,
        stiffness=0.65,
        damping=0.35,
        roughness=0.15,
        graininess=0.10,
        # Ice: muted, stick-slip squeal, mid-range modes
        modal_frequencies=(200.0, 500.0, 900.0, 1500.0),
        modal_weights=    (0.18,  0.22,  0.15,  0.10),
        modal_decay=      (0.12,  0.08,  0.05,  0.035),
        noise_color="pink",
    ),

    MAT_SNOW: MaterialAcousticProfile(
        density=0.25,
        stiffness=0.15,
        damping=0.90,
        roughness=0.40,
        graininess=0.70,
        # Snow: suppressed low spectrum, strong damping, creak on shear
        modal_frequencies=(60.0,  130.0, 260.0),
        modal_weights=    (0.12,  0.08,  0.05),
        modal_decay=      (0.02,  0.015, 0.010),
        noise_color="brown",
    ),

    MAT_PLATE: MaterialAcousticProfile(
        density=0.90,
        stiffness=0.85,
        damping=0.15,
        roughness=0.35,
        graininess=0.10,
        # Rock plate: very low infrasound modes, very long decay
        modal_frequencies=(12.0,  25.0, 42.0, 68.0),
        modal_weights=    (0.40,  0.30, 0.20, 0.10),
        modal_decay=      (1.20,  0.90, 0.60, 0.40),
        noise_color="brown",
    ),
}

# Fall-back for unknown IDs
_FALLBACK = _PROFILES[MAT_BASALT]


# ---------------------------------------------------------------------------
# MaterialAcousticDB
# ---------------------------------------------------------------------------

class MaterialAcousticDB:
    """Look-up table that maps material IDs to :class:`MaterialAcousticProfile`.

    The built-in entries can be extended at runtime via :meth:`register`.
    """

    def __init__(self) -> None:
        self._db: Dict[int, MaterialAcousticProfile] = dict(_PROFILES)

    def get(self, mat_id: int) -> MaterialAcousticProfile:
        """Return the acoustic profile for *mat_id*.  Falls back to basalt."""
        return self._db.get(mat_id, _FALLBACK)

    def register(self, mat_id: int, profile: MaterialAcousticProfile) -> None:
        """Register (or replace) a profile for *mat_id*."""
        self._db[mat_id] = profile

    @staticmethod
    def mix(
        a: MaterialAcousticProfile,
        b: MaterialAcousticProfile,
        ratio: float,
    ) -> MaterialAcousticProfile:
        """Linear-interpolate two profiles.

        Parameters
        ----------
        ratio : float
            0.0 → pure *a*, 1.0 → pure *b*.
        """
        r = max(0.0, min(1.0, ratio))
        q = 1.0 - r

        def _lerp(x: float, y: float) -> float:
            return x * q + y * r

        # Harmonise modal arrays to the same length by padding with zeros.
        len_a = len(a.modal_frequencies)
        len_b = len(b.modal_frequencies)
        n = max(len_a, len_b)

        def _pad(t: tuple, length: int, fill: float = 0.0) -> List[float]:
            lst = list(t)
            while len(lst) < length:
                lst.append(fill)
            return lst

        freqs_a   = _pad(a.modal_frequencies, n, fill=a.modal_frequencies[-1] if a.modal_frequencies else 0.0)
        freqs_b   = _pad(b.modal_frequencies, n, fill=b.modal_frequencies[-1] if b.modal_frequencies else 0.0)
        weights_a = _pad(a.modal_weights,     n, fill=0.0)
        weights_b = _pad(b.modal_weights,     n, fill=0.0)
        decay_a   = _pad(a.modal_decay,       n, fill=a.modal_decay[-1] if a.modal_decay else 0.01)
        decay_b   = _pad(b.modal_decay,       n, fill=b.modal_decay[-1] if b.modal_decay else 0.01)

        mixed_freqs   = tuple(_lerp(fa, fb) for fa, fb in zip(freqs_a,   freqs_b))
        mixed_weights = tuple(_lerp(wa, wb) for wa, wb in zip(weights_a, weights_b))
        mixed_decay   = tuple(_lerp(da, db) for da, db in zip(decay_a,   decay_b))

        noise_color = a.noise_color if ratio < 0.5 else b.noise_color

        return MaterialAcousticProfile(
            density=_lerp(a.density,    b.density),
            stiffness=_lerp(a.stiffness, b.stiffness),
            damping=_lerp(a.damping,    b.damping),
            roughness=_lerp(a.roughness, b.roughness),
            graininess=_lerp(a.graininess, b.graininess),
            modal_frequencies=mixed_freqs,
            modal_weights=mixed_weights,
            modal_decay=mixed_decay,
            noise_color=noise_color,
        )
