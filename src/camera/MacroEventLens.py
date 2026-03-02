"""MacroEventLens — Stage 41 camera reactions to macro / mega events (section 9).

Translates macro-event proximity and type into a :class:`CameraIntent`
modifier.  The modifier is *additive* on top of the normal bias produced by
:class:`~src.camera.CinematicBias.CinematicBias`.

Event reactions (section 9):

9.1 Rift nearby (``is_rift=True``)
  * slow pullback (distanceScale increases)
  * tilt toward epicenter
  * FOV slightly wider

9.2 Dust wall on horizon (``is_dust_wall=True``)
  * camera slightly higher (more sky in frame)
  * slight pullback

Public API
----------
MacroEventInput (dataclass)
MacroEventLens
  .compute(macro) → CameraIntent
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.math.Vec3 import Vec3
from src.camera.CameraIntent import CameraIntent


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class MacroEventInput:
    """Macro / mega event data consumed by :class:`MacroEventLens`.

    Attributes
    ----------
    proximity :
        0 = distant / inactive, 1 = very close / active.
    epicenter :
        World-space position of the event epicenter (rift centre, storm eye).
    is_rift :
        True for GREAT_RIFT type events (section 9.1).
    is_dust_wall :
        True for approaching SUPERCELL / DUST_WALL phenomena (section 9.2).
    """

    proximity: float = 0.0
    epicenter: Vec3 = field(default_factory=Vec3.zero)
    is_rift: bool = False
    is_dust_wall: bool = False


class MacroEventLens:
    """Translates macro-event proximity into CameraIntent modifications.

    The resulting intent is merged additively by the caller
    (CinematicBias).  All values are proportional to *proximity* so the
    transition is smooth and deterministic.
    """

    def compute(self, macro: MacroEventInput) -> CameraIntent:
        """Return a :class:`CameraIntent` delta for the given macro event."""
        intent = CameraIntent()
        p = _clamp(macro.proximity, 0.0, 1.0)
        if p <= 0.0:
            return intent

        if macro.is_rift:
            # Section 9.1: slow pullback + tilt toward epicenter + wider FOV
            intent.distance_scale = 1.0 + p * 1.0    # up to 2× distance
            intent.fov_bias = p * 5.0                 # up to +5°
            intent.height_bias = -p * 0.3             # slightly lower
            intent.tilt_bias = p * 4.0                # tilt toward epicenter
        elif macro.is_dust_wall:
            # Section 9.2: higher camera → more sky in frame
            intent.height_bias = p * 1.2              # up to +1.2 m
            intent.distance_scale = 1.0 + p * 0.4    # slight pullback
            intent.fov_bias = p * 2.0
        else:
            # Generic macro event: modest pullback + FOV expansion
            intent.distance_scale = 1.0 + p * 0.5
            intent.fov_bias = p * 3.0

        return intent
