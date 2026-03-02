"""ShakeModel — Stage 41 deterministic low-frequency camera shake (section 14).

Design principles (section 14):
* No ``random()`` — uses seeded Perlin-like sine-product noise.
* Seeded from ``player_id + tick_bucket`` for determinism across restarts.
* Frequency limited to < 3–4 Hz (section 12 anti-nausea requirement).
* Amplitude driven by vibrationLevel, constraintForce, landingImpulse.

Public API
----------
ShakeModel(player_id=0)
  .update(dt)                               — advance internal timer
  .compute(vibration_level, constraint_force, landing_impulse,
           up, forward) → (Vec3 offset, float intensity)
"""
from __future__ import annotations

import math

from src.math.Vec3 import Vec3


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _noise(x: float, y: float, seed: int) -> float:
    """Deterministic Perlin-like noise in [-1, 1].

    Uses sine / cosine products seeded by a scalar derived from *seed*.
    No ``random()`` — output is fully deterministic for given inputs.
    """
    s = float(seed & 0xFFFF) * 1.618033
    return math.sin(x * 7.3 + s) * math.cos(y * 5.1 + s * 0.7)


class ShakeModel:
    """Deterministic, low-frequency camera shake (section 14).

    Shake frequency is kept below :attr:`SHAKE_FREQ_HZ` (< 3 Hz) to
    prevent motion sickness (section 12).  The tick-bucket seed changes
    at most once per ``1 / SHAKE_FREQ_HZ`` seconds, so the noise band is
    narrow and smooth.
    """

    SHAKE_FREQ_HZ: float = 2.5      # < 3 Hz — safe threshold (section 12)
    _VERTICAL_WEIGHT: float = 0.4   # vertical noise weight vs lateral
    _OFFSET_SCALE: float = 0.05     # metres per unit amplitude

    def __init__(self, player_id: int = 0) -> None:
        self._player_id = player_id
        self._time: float = 0.0

    def update(self, dt: float) -> None:
        """Advance internal time accumulator."""
        self._time += dt

    def compute(
        self,
        vibration_level: float,
        constraint_force: float,
        landing_impulse: float,
        up: Vec3,
        forward: Vec3,
    ) -> tuple:
        """Return ``(world-space shake offset, scalar intensity 0–1)``.

        Parameters
        ----------
        vibration_level :
            From PerceptionState.vibrationLevel [0..1].
        constraint_force :
            Normalised grasp constraint tension [0..1] (section 14).
        landing_impulse :
            One-frame landing spike [0..1] (section 14).
        up :
            Planet up vector at character position.
        forward :
            Camera / character forward tangent direction.
        """
        amplitude = _clamp(
            vibration_level * 0.6 + constraint_force * 0.3 + landing_impulse * 0.1,
            0.0,
            1.0,
        )

        if amplitude < 1e-6:
            return Vec3.zero(), 0.0

        # Tick-bucket seeding (section 14: seeded from playerId + tickBucket)
        tick_bucket = int(self._time * self.SHAKE_FREQ_HZ)
        seed = (self._player_id * 1009 + tick_bucket) & 0xFFFFFF

        # Two orthogonal noise channels (lateral + vertical)
        n1 = _noise(self._time * 1.8, 0.0, seed)         # lateral
        n2 = _noise(0.0, self._time * 2.1, seed + 7)     # vertical (half weight)

        right = up.cross(forward)
        if right.length_sq() < 1e-10:
            right = Vec3(1.0, 0.0, 0.0)
        right = right.normalized()

        # Scale: shake_k is small so offset stays sub-centimetre at low amplitude
        offset = (right * n1 + up * (n2 * self._VERTICAL_WEIGHT)) * amplitude * self._OFFSET_SCALE
        return offset, amplitude
