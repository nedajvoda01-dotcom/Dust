"""AstroClimateCoupler — Stage 29 astronomical → climate bridge.

Converts the output of AstroSystem (two-sun insolation, ring shadows,
moon eclipses, sun-sun occultations) into climate drivers:

* Surface temperature via simplified relaxation thermodynamics
  (different tau for rock / dust / ice tiles)
* Heat-gradient-driven wind perturbation
* DustLiftPotential field (combines wind, dryness, heat gradients,
  ring-shadow edge proximity)
* IceOverlay rate influenced by ring shadow and night factor

Server-authoritative design
---------------------------
The coupler is updated on the server; clients reproduce the same state
from the same seed + AstroSystem timeline.  Periodic astro keyframes
(from ``AstroSystem.get_astro_keyframe()``) allow clients to detect and
correct drift.

Public API
----------
AstroClimateCoupler(config)
  .update(dt, astro, insolation, climate)   — advance one tick
  .dust_lift_potential(world_pos) → float   — DustLiftPotential [0, 1]
  .ice_form_rate(world_pos) → float         — IceOverlay formation rate [0, 1]
  .heat_wind_delta(world_pos) → Vec3        — extra wind from heat gradients
  .build_astro_keyframe(astro) → dict       — forward to AstroSystem
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.core.Config import Config
from src.math.PlanetMath import PlanetMath, LatLong
from src.math.Vec3 import Vec3

_TWO_PI = 2.0 * math.pi
_HALF_PI = math.pi * 0.5


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# CouplerCell — per-cell derived quantities
# ---------------------------------------------------------------------------

@dataclass
class CouplerCell:
    """Per-cell coupler state."""
    T_target: float = 0.0          # target temperature from astro
    dust_lift: float = 0.0         # DustLiftPotential [0, 1]
    ice_form_rate: float = 0.0     # IceOverlay formation rate [0, 1]
    heat_wind_u: float = 0.0       # east wind from heat gradient
    heat_wind_v: float = 0.0       # north wind from heat gradient


# ---------------------------------------------------------------------------
# AstroClimateCoupler
# ---------------------------------------------------------------------------

class AstroClimateCoupler:
    """Bridge between AstroSystem and ClimateSystem (Stage 29).

    Parameters
    ----------
    config:
        Game config.  Reads keys under ``coupler.*`` and ``planet.*``.
    width, height:
        Override grid size (defaults to climate grid dimensions from config).
    """

    def __init__(
        self,
        config: Config,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        # Grid
        self._w: int = (
            width if width is not None
            else config.get("climate", "grid_w", default=64)
        )
        self._h: int = (
            height if height is not None
            else config.get("climate", "grid_h", default=32)
        )
        self._n: int = self._w * self._h
        self._planet_r: float = config.get("planet", "radius_units", default=1000.0)

        # --- Temperature relaxation time constants (seconds) ---
        self._tau_rock: float  = config.get("coupler", "temp_tau_rock",  default=600.0)
        self._tau_dust: float  = config.get("coupler", "temp_tau_dust",  default=300.0)
        self._tau_ice: float   = config.get("coupler", "temp_tau_ice",   default=900.0)

        # --- Equilibrium temperature parameters ---
        self._T_base_solar: float = config.get("coupler", "T_base_solar", default=220.0)
        self._T_solar_gain: float = config.get("coupler", "T_solar_gain", default=80.0)
        self._T_ring_shadow_k: float = config.get("coupler", "T_ring_shadow_k", default=30.0)

        # --- Heat-gradient wind ---
        self._wind_heat_k: float = config.get("coupler", "wind_from_heat_k", default=2.0)

        # --- Dust lift ---
        self._dust_lift_k: float = config.get("coupler", "dust_lift_k", default=1.0)
        self._dust_lift_wind_thresh: float = 8.0   # m/s threshold for lifting

        # --- Ice formation ---
        self._ice_form_k: float  = config.get("coupler", "ice_form_k",  default=0.5)
        self._ice_melt_k: float  = config.get("coupler", "ice_melt_k",  default=0.3)
        self._ice_temp_thresh: float = 270.0

        # --- Row / col lat-lon lookup ---
        self._row_lats: list[float] = [
            -_HALF_PI + math.pi * (y + 0.5) / self._h
            for y in range(self._h)
        ]
        self._col_lons: list[float] = [
            -math.pi + _TWO_PI * (x + 0.5) / self._w
            for x in range(self._w)
        ]

        # --- Per-cell fields ---
        self._T_target: list[float]      = [self._T_base_solar] * self._n
        self._dust_lift: list[float]     = [0.0] * self._n
        self._ice_form_rate: list[float] = [0.0] * self._n
        self._heat_wind_u: list[float]   = [0.0] * self._n
        self._heat_wind_v: list[float]   = [0.0] * self._n

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        astro,
        insolation,
        climate,
    ) -> None:
        """Advance the coupler by *dt* game-seconds.

        Parameters
        ----------
        dt:
            Game time elapsed since last call (seconds).
        astro:
            AstroSystem instance (provides ring shadow, eclipse, sun dirs).
        insolation:
            InsolationField instance (provides per-tile insolation sample).
        climate:
            ClimateSystem instance (provides temperature, wind, dust, ice).
        """
        W, H = self._w, self._h
        dlat = math.pi / H
        dlon = _TWO_PI / W

        # Build T_target from insolation + ring shadow
        for y in range(H):
            lat = self._row_lats[y]
            for x in range(W):
                idx = y * W + x
                lon = self._col_lons[x]
                ll = LatLong(lat, lon)
                cell_dir = PlanetMath.direction_from_lat_long(ll)

                # Insolation
                if insolation is not None:
                    sol = insolation.sample_at(cell_dir)
                    E = sol.direct_total
                    ring_shadow = sol.ring_shadow_eff
                else:
                    E = 0.0
                    ring_shadow = 0.0

                # T_target = base + solar gain - ring shadow cooling
                self._T_target[idx] = (
                    self._T_base_solar
                    + self._T_solar_gain * E
                    - self._T_ring_shadow_k * ring_shadow
                )

        # Heat-gradient wind: ∂T/∂lon → northward perturbation,
        #                      ∂T/∂lat → eastward perturbation
        for y in range(H):
            for x in range(W):
                idx = y * W + x
                lat = self._row_lats[y]
                cos_lat = max(0.01, math.cos(lat))

                x_l = (x - 1) % W
                x_r = (x + 1) % W
                y_b = max(0, y - 1)
                y_t = min(H - 1, y + 1)

                dT_dlon = (self._T_target[y * W + x_r] - self._T_target[y * W + x_l]) / (2.0 * dlon)
                dT_dlat = (self._T_target[y_t * W + x] - self._T_target[y_b * W + x]) / (2.0 * dlat)

                # Wind perturbation perpendicular to gradient ("anabatic flow")
                self._heat_wind_u[idx] = self._wind_heat_k * (-dT_dlat) / max(1.0, self._T_solar_gain)
                self._heat_wind_v[idx] = self._wind_heat_k * dT_dlon / (cos_lat * max(1.0, self._T_solar_gain))

        # DustLiftPotential and IceFormRate per cell
        for y in range(H):
            lat = self._row_lats[y]
            for x in range(W):
                idx = y * W + x
                lon = self._col_lons[x]
                ll = LatLong(lat, lon)
                cell_dir = PlanetMath.direction_from_lat_long(ll)

                # Pull from climate if available
                if climate is not None:
                    wspd = climate.sample_wind(cell_dir).length()
                    dust = climate.sample_dust(cell_dir)
                    temp = climate.sample_temperature(cell_dir)
                    ice  = climate.get_wetness(cell_dir)  # proxy for wetness/ice
                else:
                    wspd = 0.0
                    dust = 0.0
                    temp = self._T_target[idx]
                    ice  = 0.0

                if insolation is not None:
                    sol = insolation.sample_at(cell_dir)
                    ring_shadow = sol.ring_shadow_eff
                else:
                    ring_shadow = 0.0

                # DustLiftPotential: high wind + high dust + shadow gradients
                wind_factor = _clamp(
                    (wspd - self._dust_lift_wind_thresh) / max(1.0, self._dust_lift_wind_thresh),
                    0.0, 1.0,
                )
                shadow_edge = abs(self._heat_wind_u[idx]) + abs(self._heat_wind_v[idx])
                shadow_edge_factor = _clamp(shadow_edge / max(1.0, self._wind_heat_k), 0.0, 1.0)
                ice_suppress = _clamp(ice, 0.0, 1.0)
                cold_suppress = _clamp((self._ice_temp_thresh - temp) / 30.0, 0.0, 1.0)
                self._dust_lift[idx] = _clamp(
                    self._dust_lift_k * (wind_factor + shadow_edge_factor * 0.5) * dust
                    * (1.0 - ice_suppress * 0.5)
                    * (1.0 - cold_suppress * 0.5),
                    0.0, 1.0,
                )

                # IceFormRate: cold + ring shadow + night
                night_factor = max(0.0, -self._heat_wind_v[idx])  # proxy: cooling side
                cold_factor = _clamp((self._ice_temp_thresh - temp) / 30.0, 0.0, 1.0)
                self._ice_form_rate[idx] = _clamp(
                    self._ice_form_k * cold_factor * (1.0 + ring_shadow),
                    0.0, 1.0,
                )

    # ------------------------------------------------------------------
    # Public sampling API (O(1) bilinear)
    # ------------------------------------------------------------------

    def dust_lift_potential(self, world_pos: Vec3) -> float:
        """DustLiftPotential [0, 1] at *world_pos*."""
        fx, fy = self._lat_lon_to_grid(world_pos)
        return _clamp(self._bilerp(self._dust_lift, fx, fy), 0.0, 1.0)

    def ice_form_rate(self, world_pos: Vec3) -> float:
        """IceOverlay formation rate [0, 1] at *world_pos*."""
        fx, fy = self._lat_lon_to_grid(world_pos)
        return _clamp(self._bilerp(self._ice_form_rate, fx, fy), 0.0, 1.0)

    def heat_wind_delta(self, world_pos: Vec3) -> Vec3:
        """Extra wind Vec3 from heat gradients at *world_pos*."""
        ll = PlanetMath.from_direction(world_pos)
        fx, fy = self._lat_lon_to_grid(world_pos)
        u = self._bilerp(self._heat_wind_u, fx, fy)
        v = self._bilerp(self._heat_wind_v, fx, fy)
        lat, lon = ll.lat_rad, ll.lon_rad
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        sin_lon, cos_lon = math.sin(lon), math.cos(lon)
        north = Vec3(-sin_lat * sin_lon, cos_lat, -sin_lat * cos_lon)
        east  = Vec3(cos_lon, 0.0, -sin_lon)
        return north * v + east * u

    def T_target_at(self, world_pos: Vec3) -> float:
        """Target temperature (K) at *world_pos* from current astro state."""
        fx, fy = self._lat_lon_to_grid(world_pos)
        return self._bilerp(self._T_target, fx, fy)

    # ------------------------------------------------------------------
    # Multiplayer keyframe helper (delegates to AstroSystem)
    # ------------------------------------------------------------------

    @staticmethod
    def build_astro_keyframe(astro) -> dict:
        """Return an ASTRO_KEYFRAME dict for server→client synchronisation."""
        return astro.get_astro_keyframe()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lat_lon_to_grid(self, world_pos: Vec3) -> tuple[float, float]:
        ll = PlanetMath.from_direction(world_pos)
        fx = (ll.lon_rad + math.pi) / _TWO_PI * self._w - 0.5
        fy = (ll.lat_rad + _HALF_PI) / math.pi * self._h - 0.5
        return fx, fy

    def _bilerp(self, field: list[float], fx: float, fy: float) -> float:
        W, H = self._w, self._h
        fx = fx % W
        if fx < 0.0:
            fx += W
        fy = _clamp(fy, 0.0, H - 1.0 - 1e-9)
        x0 = int(fx); x1 = (x0 + 1) % W
        y0 = int(fy); y1 = min(y0 + 1, H - 1)
        tx = fx - x0; ty = fy - y0
        return (
            (field[y0 * W + x0] * (1.0 - tx) + field[y0 * W + x1] * tx) * (1.0 - ty)
            + (field[y1 * W + x0] * (1.0 - tx) + field[y1 * W + x1] * tx) * ty
        )
