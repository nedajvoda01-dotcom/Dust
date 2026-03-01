"""CharacterSpawnSystem — Stage 20 safe spawn-point selection.

Deterministically selects a spawn location on the planet surface that
avoids steep slopes, unstable geology, and severe storm conditions.

Algorithm (Section 5 of Stage 20 spec)
---------------------------------------
1. Derive a base lat/lon from the seed (deterministic).
2. Generate up to *spawn_attempts* candidates scattered around the base.
3. Score each candidate:
     + flat terrain  (slope below threshold)
     + stable ground (low fracture / stress)
     + good visibility (not in full whiteout)
     − steep slope, instability, storm
4. Return the highest-scoring candidate.
5. Fallback: if no candidate scores positively use the best available;
   if still nothing use the planet origin-up direction.

Public API
----------
CharacterSpawnSystem(seed, planet_radius, ...)
  .spawn(saved_lat=None, saved_lon=None) → SpawnCandidate
  .get_spawn_world_pos(candidate) → Vec3
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from src.core.Logger import Logger
from src.math.PlanetMath import PlanetMath, LatLong
from src.math.Vec3 import Vec3

_TAG = "SpawnSystem"

# Seed mixer so the spawn RNG doesn't overlap with other systems' seeds.
_SPAWN_SEED_MIX = 0x5A291AB0

# Approximate character eye height above the ground surface (simulation units).
# Matches the default capsule_height in CharacterPhysicalController / char config.
_CAPSULE_HEIGHT_OFFSET = 1.8


@dataclass
class SpawnCandidate:
    """A candidate spawn location with its evaluated score."""
    lat_rad:  float
    lon_rad:  float
    unit_dir: Vec3
    score:    float


class CharacterSpawnSystem:
    """Selects a safe spawn point on the planet surface.

    Parameters
    ----------
    seed:
        World seed (deterministic candidate selection).
    planet_radius:
        Planet surface radius (simulation units).
    spawn_attempts:
        Number of candidate points evaluated (more → better spawn, slower).
    slope_max_deg:
        Maximum acceptable slope angle; steeper terrain is penalised heavily.
    stability_min:
        Minimum geological stability [0, 1]; below this the candidate is
        penalised.
    avoid_storm_threshold:
        Visibility threshold below which a storm-covered spawn is penalised.
    height_provider:
        Optional PlanetHeightProvider for accurate slope estimation.
    tectonic_system:
        Optional TectonicPlatesSystem for geological stability.
    climate_system:
        Optional ClimateSystem for visibility sampling.
    """

    def __init__(
        self,
        seed:                   int,
        planet_radius:          float = 1000.0,
        spawn_attempts:         int   = 64,
        slope_max_deg:          float = 25.0,
        stability_min:          float = 0.3,
        avoid_storm_threshold:  float = 0.8,
        height_provider               = None,
        tectonic_system               = None,
        climate_system                = None,
    ) -> None:
        self._rng            = random.Random((seed ^ _SPAWN_SEED_MIX) & 0xFFFFFFFF)
        self._planet_radius  = planet_radius
        self._attempts       = spawn_attempts
        self._slope_max_rad  = math.radians(slope_max_deg)
        self._stability_min  = stability_min
        self._storm_thresh   = avoid_storm_threshold
        self._height         = height_provider
        self._tectonic       = tectonic_system
        self._climate        = climate_system

        # Pick a deterministic base location from the seed.
        # Restrict to ±45° latitude for moderate-climate spawns.
        self._base_lat = (self._rng.random() - 0.5) * math.pi * 0.5
        self._base_lon = self._rng.random() * 2.0 * math.pi

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def spawn(
        self,
        saved_lat: Optional[float] = None,
        saved_lon: Optional[float] = None,
    ) -> SpawnCandidate:
        """Return the best spawn candidate.

        If *saved_lat* / *saved_lon* are provided (restored from a save)
        that position is validated first; if it scores acceptably it is
        returned without a full search.
        """
        if saved_lat is not None and saved_lon is not None:
            try:
                unit_dir = PlanetMath.direction_from_lat_long(LatLong(saved_lat, saved_lon))
                candidate = self._evaluate(unit_dir)
                if candidate.score >= 0.0:
                    Logger.info(
                        _TAG,
                        f"Restored spawn: lat={math.degrees(saved_lat):.2f}° "
                        f"lon={math.degrees(saved_lon):.2f}°",
                    )
                    return candidate
            except Exception:
                pass
            Logger.warn(_TAG, "Saved spawn invalid; searching for new safe spawn")

        return self._find_best()

    def get_spawn_world_pos(self, candidate: SpawnCandidate) -> Vec3:
        """World-space position for *candidate* (character eye height above surface)."""
        h = 0.0
        if self._height is not None:
            try:
                h = float(self._height.sample_height(candidate.unit_dir))
                if not math.isfinite(h):
                    h = 0.0
            except Exception:
                h = 0.0
        capsule_offset = _CAPSULE_HEIGHT_OFFSET
        return candidate.unit_dir * (self._planet_radius + h + capsule_offset)

    # ------------------------------------------------------------------
    # Private: search
    # ------------------------------------------------------------------

    def _find_best(self) -> SpawnCandidate:
        """Evaluate candidates and return the highest-scoring one."""
        best: Optional[SpawnCandidate] = None
        jitter_lat = math.radians(30.0)
        jitter_lon = math.radians(60.0)

        for _ in range(self._attempts):
            lat = self._base_lat + (self._rng.random() - 0.5) * jitter_lat
            lon = self._base_lon + (self._rng.random() - 0.5) * jitter_lon
            # Clamp lat to avoid poles
            lat = max(-math.pi * 0.45, min(math.pi * 0.45, lat))
            unit_dir = PlanetMath.direction_from_lat_long(LatLong(lat, lon))
            cand = self._evaluate(unit_dir)
            if best is None or cand.score > best.score:
                best = cand

        if best is None or best.score < -1.0:
            # Ultimate fallback: directly above spawn base
            unit_dir = PlanetMath.direction_from_lat_long(LatLong(self._base_lat, self._base_lon))
            best = self._evaluate(unit_dir)
            Logger.warn(_TAG, "Using fallback spawn (no high-scoring candidate found)")

        Logger.info(
            _TAG,
            f"Spawn: lat={math.degrees(best.lat_rad):.2f}° "
            f"lon={math.degrees(best.lon_rad):.2f}° "
            f"score={best.score:.3f}",
        )
        return best

    # ------------------------------------------------------------------
    # Private: scoring
    # ------------------------------------------------------------------

    def _evaluate(self, unit_dir: Vec3) -> SpawnCandidate:
        ll    = PlanetMath.from_direction(unit_dir)
        score = 0.0

        # --- Slope ---
        slope = self._estimate_slope(unit_dir)
        if slope > self._slope_max_rad:
            score -= 2.0 * (slope - self._slope_max_rad) / math.pi
        else:
            score += 0.5 * (1.0 - slope / max(self._slope_max_rad, 1e-6))

        # --- Geological stability ---
        stability = self._get_stability(unit_dir)
        if stability < self._stability_min:
            score -= (self._stability_min - stability)
        else:
            score += 0.3 * stability

        # --- Visibility (storm avoidance) ---
        visibility = self._get_visibility(unit_dir)
        if visibility < self._storm_thresh:
            score -= 0.5 * (1.0 - visibility / max(self._storm_thresh, 1e-6))
        else:
            score += 0.2 * visibility

        return SpawnCandidate(
            lat_rad  = ll.lat_rad,
            lon_rad  = ll.lon_rad,
            unit_dir = unit_dir,
            score    = score,
        )

    def _estimate_slope(self, unit_dir: Vec3) -> float:
        """Estimate local slope angle from height samples at neighbouring points."""
        if self._height is None:
            return 0.0
        try:
            h_c = float(self._height.sample_height(unit_dir))
            if not math.isfinite(h_c):
                return 0.0
            delta    = 0.005  # small angular step
            max_slope = 0.0
            for dx, dz in ((delta, 0.0), (-delta, 0.0), (0.0, delta), (0.0, -delta)):
                nb = Vec3(
                    unit_dir.x + dx,
                    unit_dir.y,
                    unit_dir.z + dz,
                ).normalized()
                h_nb = float(self._height.sample_height(nb))
                if not math.isfinite(h_nb):
                    continue
                dh   = abs(h_nb - h_c)
                dist = delta * self._planet_radius
                if dist > 0.0:
                    s = math.atan2(dh, dist)
                    if s > max_slope:
                        max_slope = s
            return max_slope
        except Exception:
            return 0.0

    def _get_stability(self, unit_dir: Vec3) -> float:
        """Geological stability [0, 1] for *unit_dir* (1 = fully stable)."""
        if self._tectonic is None:
            return 1.0
        try:
            cell = self._tectonic.sample_field_cell(unit_dir)
            if cell is not None:
                fracture = getattr(cell, "fracture", 0.0)
                stress   = getattr(cell, "stress",   0.0)
                return max(0.0, 1.0 - fracture * 0.7 - stress * 0.3)
        except Exception:
            pass
        return 1.0

    def _get_visibility(self, unit_dir: Vec3) -> float:
        """Atmospheric visibility [0, 1] at *unit_dir*."""
        if self._climate is None:
            return 1.0
        try:
            pos = unit_dir * self._planet_radius
            vis = float(self._climate.get_visibility(pos))
            if math.isfinite(vis):
                return max(0.0, min(1.0, vis))
        except Exception:
            pass
        return 1.0
