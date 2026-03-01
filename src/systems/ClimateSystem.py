"""ClimateSystem — Stage 8 planetary climate simulation.

Maintains equirectangular grid fields (temp, pressure, wind, dust, humidity,
ice) updated deterministically from insolation and seed.

Grid layout (shared with InsolationField convention):
  row 0 → south pole (−π/2 lat), row H−1 → north pole (+π/2 lat)
  col 0 → −π lon,                col W−1 → +π lon  (cell centres)

All fields stored as flat lists indexed by ``row * W + col``.

Public API
----------
update(dt, insolation=None)       — advance climate by *dt* game-seconds
sample_wind(world_pos) → Vec3     — wind tangent vector at surface position
sample_dust(world_pos) → float    — dust suspension [0, 1]
sample_temperature(world_pos) → float
get_visibility(world_pos) → float  — exp(-dust*k)
get_erosion_factor(world_pos) → float   — stub for GeoEventSystem
get_freeze_thaw_factor(world_pos) → float
get_wind_force_factor(world_pos) → float
get_wetness(world_pos) → float

wind_lat_long_to_world(lat, lon, u, v) → Vec3   — static utility
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field as dc_field

from src.core.Config import Config
from src.math.PlanetMath import PlanetMath, LatLong
from src.math.Vec3 import Vec3
from src.systems.ClimateSystemStub import IClimateSystem

_TWO_PI = 2.0 * math.pi
_HALF_PI = math.pi * 0.5


# ---------------------------------------------------------------------------
# StormCell
# ---------------------------------------------------------------------------

@dataclass
class StormCell:
    """An active dust-storm cell tracked on the sphere."""
    center_lat: float     # radians
    center_lon: float     # radians
    radius: float         # angular radius in radians (~0.2–0.5 rad)
    intensity: float      # 0–1
    vel_u: float          # eastward speed  (m/s, same units as wind fields)
    vel_v: float          # northward speed
    lifetime: float       # remaining seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _spatial_hash(x: int, y: int, tick: int) -> float:
    """Deterministic float in [0, 1) from three integers.

    Used for wind turbulence so we never draw from the seeded RNG
    (keeping that RNG purely for stochastic storm spawning).
    """
    n = x * 1731 + y * 2971 + tick * 131071
    n = (n ^ (n >> 13)) * 1664525 + 1013904223
    n = (n ^ (n >> 16)) & 0x7FFFFFFF
    return n / 2147483648.0


# ---------------------------------------------------------------------------
# ClimateSystem
# ---------------------------------------------------------------------------

class ClimateSystem(IClimateSystem):
    """Full planetary climate simulation (Stage 8).

    Parameters
    ----------
    config:
        Game config object (reads ``climate.*`` and ``planet.*`` keys).
    seed:
        Deterministic seed for all stochastic elements (storm birth).
    width, height:
        Override grid dimensions (ignores config values when provided).
    """

    def __init__(
        self,
        config: Config,
        seed: int = 42,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        # --- Grid dimensions ---
        self._w: int = width  if width  is not None else config.get("climate", "grid_w",  default=64)
        self._h: int = height if height is not None else config.get("climate", "grid_h",  default=32)
        self._n: int = self._w * self._h

        # --- Planet scale (needed for advection unit conversion) ---
        self._planet_r: float = config.get("planet", "radius_units", default=1000.0)

        # --- Climate parameters (all read from config with sensible defaults) ---
        self._temp_base_equator: float = config.get("climate", "temp_base_equator", default=290.0)
        self._temp_lat_k:        float = config.get("climate", "temp_lat_k",        default=60.0)
        self._heat_gain:         float = config.get("climate", "heat_gain",          default=30.0)
        self._heat_loss:         float = config.get("climate", "heat_loss",          default=0.1)
        self._pressure_from_temp: float = config.get("climate", "pressure_from_temp", default=0.1)
        self._wind_base_strength: float = config.get("climate", "wind_base_strength", default=4.5)
        self._wind_gradP_strength: float = config.get("climate", "wind_gradP_strength", default=0.5)
        self._wind_max:          float = config.get("climate", "wind_max",           default=40.0)
        self._wind_damping:      float = config.get("climate", "wind_damping",       default=0.02)
        self._dust_deposit_rate: float = config.get("climate", "dust_deposit_rate",  default=0.05)
        self._dust_lift_rate:    float = config.get("climate", "dust_lift_rate",     default=0.02)
        self._dust_lift_threshold: float = config.get("climate", "dust_lift_threshold", default=8.0)
        self._storm_dust_threshold: float = config.get("climate", "storm_dust_threshold", default=0.6)
        self._storm_wind_threshold: float = config.get("climate", "storm_wind_threshold", default=15.0)
        self._visibility_k:      float = config.get("climate", "visibility_k",       default=5.0)
        self._freeze_threshold:  float = config.get("climate", "freeze_threshold",   default=270.0)
        self._melt_threshold:    float = config.get("climate", "melt_threshold",     default=280.0)

        # --- Seeded RNG for stochastic events ---
        self._rng = random.Random(seed)

        # --- Per-row lat / per-col lon lookup tables ---
        self._row_lats: list[float] = [
            -_HALF_PI + math.pi * (y + 0.5) / self._h
            for y in range(self._h)
        ]
        self._col_lons: list[float] = [
            -math.pi + _TWO_PI * (x + 0.5) / self._w
            for x in range(self._w)
        ]

        # --- Climate fields (flat lists) ---
        self._temp:     list[float] = [0.0] * self._n
        self._pressure: list[float] = [0.0] * self._n
        self._wind_u:   list[float] = [0.0] * self._n   # eastward  (m/s)
        self._wind_v:   list[float] = [0.0] * self._n   # northward (m/s)
        self._dust:     list[float] = [0.0] * self._n
        self._humidity: list[float] = [0.0] * self._n
        self._ice:      list[float] = [0.0] * self._n

        # --- Storm cells ---
        self._storms: list[StormCell] = []
        self._storm_check_counter: int = 0
        self._storm_check_interval: int = 10   # check every N ticks

        # --- Update tick counter (for turbulence hash) ---
        self._tick: int = 0

        # --- Initialise fields ---
        self._init_fields()

    # ------------------------------------------------------------------
    # IClimateSystem interface
    # ------------------------------------------------------------------

    def update(self, dt: float, insolation=None) -> None:
        """Advance climate simulation by *dt* game-seconds.

        Parameters
        ----------
        dt:
            Game-time seconds elapsed since last call.
        insolation:
            Optional ``InsolationField``.  When provided, temperature
            responds to solar irradiance; otherwise only latitude-based
            relaxation applies.
        """
        self._update_temperature(dt, insolation)
        self._update_pressure()
        self._update_wind(dt)
        self._update_dust(dt)
        self._update_humidity_ice(dt)
        self._update_storms(dt)
        self._tick += 1

    def sample_wind(self, world_pos: Vec3) -> Vec3:
        """Wind vector at *world_pos* in the same units as wind fields (m/s)."""
        ll = PlanetMath.from_direction(world_pos)
        fx, fy = self._lat_lon_to_grid(ll.lat_rad, ll.lon_rad)
        u = self._bilerp(self._wind_u, fx, fy)
        v = self._bilerp(self._wind_v, fx, fy)
        return self.wind_lat_long_to_world(ll.lat_rad, ll.lon_rad, u, v)

    def sample_dust(self, world_pos: Vec3) -> float:
        """Dust suspension [0, 1] at *world_pos*."""
        ll = PlanetMath.from_direction(world_pos)
        fx, fy = self._lat_lon_to_grid(ll.lat_rad, ll.lon_rad)
        return _clamp(self._bilerp(self._dust, fx, fy), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Extended API (Stage 8)
    # ------------------------------------------------------------------

    def sample_temperature(self, world_pos: Vec3) -> float:
        """Surface temperature (K) at *world_pos*."""
        ll = PlanetMath.from_direction(world_pos)
        fx, fy = self._lat_lon_to_grid(ll.lat_rad, ll.lon_rad)
        return self._bilerp(self._temp, fx, fy)

    def get_visibility(self, world_pos: Vec3) -> float:
        """Visibility factor [0, 1].  1 = clear sky, 0 = complete whiteout."""
        d = self.sample_dust(world_pos)
        return math.exp(-d * self._visibility_k)

    def get_wind_force_factor(self, world_pos: Vec3) -> float:
        """Normalised wind-force factor [0, 1] for character physics."""
        speed = self.sample_wind(world_pos).length()
        return _clamp(speed / max(1.0, self._wind_max), 0.0, 1.0)

    def get_wetness(self, world_pos: Vec3) -> float:
        """Surface wetness [0, 1] from humidity + ice melt potential."""
        ll = PlanetMath.from_direction(world_pos)
        fx, fy = self._lat_lon_to_grid(ll.lat_rad, ll.lon_rad)
        h = _clamp(self._bilerp(self._humidity, fx, fy), 0.0, 1.0)
        i = _clamp(self._bilerp(self._ice,      fx, fy), 0.0, 1.0)
        return _clamp(h + i * 0.5, 0.0, 1.0)

    def get_erosion_factor(self, world_pos: Vec3) -> float:
        """Climate-driven erosion stress [0, 1] for GeoEventSystem (stub)."""
        speed = self.sample_wind(world_pos).length()
        dust  = self.sample_dust(world_pos)
        return _clamp((speed / max(1.0, self._wind_max)) * dust, 0.0, 1.0)

    def get_freeze_thaw_factor(self, world_pos: Vec3) -> float:
        """Freeze-thaw cycle intensity [0, 1] for GeoEventSystem (stub)."""
        ll = PlanetMath.from_direction(world_pos)
        fx, fy = self._lat_lon_to_grid(ll.lat_rad, ll.lon_rad)
        t   = self._bilerp(self._temp, fx, fy)
        ice = _clamp(self._bilerp(self._ice, fx, fy), 0.0, 1.0)
        mid = (self._freeze_threshold + self._melt_threshold) * 0.5
        # Factor peaks when temperature is near the freeze/melt boundary
        dist = abs(t - mid)
        proximity = _clamp(1.0 - dist / 30.0, 0.0, 1.0)
        return _clamp(ice * proximity, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Static utility
    # ------------------------------------------------------------------

    @staticmethod
    def wind_lat_long_to_world(lat: float, lon: float, u: float, v: float) -> Vec3:
        """Convert lat/lon tangent wind (u=east, v=north) to world Vec3.

        Uses the Y-up sphere convention of PlanetMath:
            P = (cos_lat·sin_lon, sin_lat, cos_lat·cos_lon)
        North tangent: dP/dlat = (−sin_lat·sin_lon, cos_lat, −sin_lat·cos_lon)
        East  tangent: dP/dlon / cos_lat = (cos_lon, 0, −sin_lon)
        """
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        sin_lon, cos_lon = math.sin(lon), math.cos(lon)

        north = Vec3(-sin_lat * sin_lon,  cos_lat, -sin_lat * cos_lon)
        # east tangent is well-defined everywhere (does not depend on cos_lat)
        east  = Vec3(cos_lon, 0.0, -sin_lon)

        return north * v + east * u

    # ------------------------------------------------------------------
    # Accessor for storm list (read-only view for dev visualisation)
    # ------------------------------------------------------------------

    @property
    def storms(self) -> list[StormCell]:
        return list(self._storms)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_fields(self) -> None:
        """Set physically plausible initial state."""
        W, H = self._w, self._h
        for y in range(H):
            lat = self._row_lats[y]
            base_t = self._temp_base_equator - self._temp_lat_k * abs(lat) / _HALF_PI
            u_base = self._wind_base_u(lat)
            for x in range(W):
                idx = y * W + x
                self._temp[idx] = base_t
                self._wind_u[idx] = u_base
                self._wind_v[idx] = 0.0
                # Small seed-dependent initial dust
                self._dust[idx] = _clamp(
                    self._rng.uniform(0.02, 0.20) * max(0.1, 1.0 - abs(lat) / _HALF_PI),
                    0.0, 1.0,
                )
        self._update_pressure()

    # ------------------------------------------------------------------
    # Temperature update
    # ------------------------------------------------------------------

    def _update_temperature(self, dt: float, insolation) -> None:
        W = self._w
        for y in range(self._h):
            lat = self._row_lats[y]
            T_base = self._temp_base_equator - self._temp_lat_k * abs(lat) / _HALF_PI
            for x in range(W):
                idx = y * W + x
                if insolation is not None:
                    ll   = LatLong(lat, self._col_lons[x])
                    cdir = PlanetMath.direction_from_lat_long(ll)
                    sol  = insolation.sample_at(cdir)
                    S    = sol.direct_total
                else:
                    S = 0.0
                T = self._temp[idx]
                T += dt * (self._heat_gain * S - self._heat_loss * (T - T_base))
                self._temp[idx] = _clamp(T, -150.0, 500.0)

    # ------------------------------------------------------------------
    # Pressure update
    # ------------------------------------------------------------------

    def _update_pressure(self) -> None:
        n = self._n
        mean_t = sum(self._temp) / n
        for idx in range(n):
            self._pressure[idx] = 1.0 + self._pressure_from_temp * (
                self._temp[idx] - mean_t
            )

    # ------------------------------------------------------------------
    # Wind update
    # ------------------------------------------------------------------

    def _update_wind(self, dt: float) -> None:
        W, H  = self._w, self._h
        dlon  = _TWO_PI / W
        dlat  = math.pi / H
        inv2dlon = 1.0 / (2.0 * dlon)
        inv2dlat = 1.0 / (2.0 * dlat)

        new_u = list(self._wind_u)
        new_v = list(self._wind_v)

        for y in range(H):
            lat = self._row_lats[y]
            u_base = self._wind_base_u(lat)
            for x in range(W):
                idx = y * W + x

                # Pressure gradient (central differences, lon wraps)
                x_l = (x - 1) % W
                x_r = (x + 1) % W
                y_b = max(0, y - 1)
                y_t = min(H - 1, y + 1)
                dP_dlon = (self._pressure[y * W + x_r] - self._pressure[y * W + x_l]) * inv2dlon
                dP_dlat = (self._pressure[y_t * W + x] - self._pressure[y_b * W + x]) * inv2dlat

                # Turbulence (deterministic spatial hash, no RNG draw)
                th = _spatial_hash(x, y, self._tick)
                turb_u = (th * 2.0 - 1.0) * 0.3
                turb_v = (_spatial_hash(x + 1000, y + 500, self._tick) * 2.0 - 1.0) * 0.2

                # Target wind = base circulation + pressure gradient + turbulence
                target_u = u_base - self._wind_gradP_strength * dP_dlon + turb_u
                target_v =        - self._wind_gradP_strength * dP_dlat + turb_v

                # Relax toward target
                relax = _clamp(0.05 * dt, 0.0, 1.0)
                u = self._wind_u[idx] + relax * (target_u - self._wind_u[idx])
                v = self._wind_v[idx] + relax * (target_v - self._wind_v[idx])

                # Damping ("friction")
                damp = _clamp(self._wind_damping * dt, 0.0, 1.0)
                u *= 1.0 - damp
                v *= 1.0 - damp

                # Speed cap
                mag = math.sqrt(u * u + v * v)
                if mag > self._wind_max:
                    scale = self._wind_max / mag
                    u *= scale
                    v *= scale

                new_u[idx] = u
                new_v[idx] = v

        self._wind_u = new_u
        self._wind_v = new_v

    # ------------------------------------------------------------------
    # Dust update (semi-Lagrangian advection + deposition/lifting)
    # ------------------------------------------------------------------

    def _update_dust(self, dt: float) -> None:
        W, H = self._w, self._h
        R    = self._planet_r
        # Semi-Lagrangian advection: back-trace one step
        new_dust: list[float] = [0.0] * self._n
        for y in range(H):
            lat = self._row_lats[y]
            cos_lat = max(0.01, math.cos(lat))
            for x in range(W):
                idx    = y * W + x
                u      = self._wind_u[idx]
                v      = self._wind_v[idx]
                lon    = self._col_lons[x]

                lat_back = lat - (v / R) * dt
                lon_back = lon - (u / (R * cos_lat)) * dt

                fx, fy = self._lat_lon_to_grid(lat_back, lon_back)
                new_dust[idx] = self._bilerp(self._dust, fx, fy)

        # Deposition and lifting
        for y in range(H):
            lat = self._row_lats[y]
            for x in range(W):
                idx  = y * W + x
                u    = self._wind_u[idx]
                v    = self._wind_v[idx]
                wspd = math.sqrt(u * u + v * v)

                deposit = self._dust_deposit_rate * new_dust[idx] * dt
                new_dust[idx] = max(0.0, new_dust[idx] - deposit)

                lift_excess = max(0.0, wspd - self._dust_lift_threshold)
                lift = self._dust_lift_rate * lift_excess * dt
                new_dust[idx] = min(1.0, new_dust[idx] + lift)

        self._dust = new_dust

    # ------------------------------------------------------------------
    # Humidity / ice update
    # ------------------------------------------------------------------

    def _update_humidity_ice(self, dt: float) -> None:
        for idx in range(self._n):
            t = self._temp[idx]
            if t < self._freeze_threshold:
                rate  = 0.01 * (self._freeze_threshold - t) / 30.0
                delta = min(self._humidity[idx], rate * dt)
                self._humidity[idx] -= delta
                self._ice[idx]       = min(1.0, self._ice[idx] + delta)
            elif t > self._melt_threshold and self._ice[idx] > 0.0:
                rate  = 0.02 * (t - self._melt_threshold) / 30.0
                delta = min(self._ice[idx], rate * dt)
                self._ice[idx]       -= delta
                self._humidity[idx]   = min(1.0, self._humidity[idx] + delta)

    # ------------------------------------------------------------------
    # Storm cells
    # ------------------------------------------------------------------

    def _update_storms(self, dt: float) -> None:
        R = self._planet_r

        # Apply storm influence on dust and wind
        for storm in self._storms:
            for y in range(self._h):
                lat = self._row_lats[y]
                for x in range(self._w):
                    lon  = self._col_lons[x]
                    dist = _angular_dist(lat, lon, storm.center_lat, storm.center_lon)
                    if dist >= storm.radius:
                        continue
                    frac = 1.0 - dist / storm.radius
                    idx  = y * self._w + x
                    # Boost dust
                    self._dust[idx] = min(1.0, self._dust[idx] + 0.1 * storm.intensity * frac * dt)
                    # Boost wind magnitude toward storm velocity
                    boost = storm.intensity * frac * dt * 0.5
                    self._wind_u[idx] = _clamp(
                        self._wind_u[idx] + (storm.vel_u - self._wind_u[idx]) * boost,
                        -self._wind_max, self._wind_max,
                    )
                    self._wind_v[idx] = _clamp(
                        self._wind_v[idx] + (storm.vel_v - self._wind_v[idx]) * boost,
                        -self._wind_max, self._wind_max,
                    )

        # Move and decay storms
        for storm in self._storms:
            cos_lat = max(0.01, math.cos(storm.center_lat))
            storm.center_lat += (storm.vel_v / R) * dt
            storm.center_lon += (storm.vel_u / (R * cos_lat)) * dt
            # Wrap / clamp
            storm.center_lon = ((storm.center_lon + math.pi) % _TWO_PI) - math.pi
            storm.center_lat = _clamp(storm.center_lat, -_HALF_PI + 0.01, _HALF_PI - 0.01)
            storm.lifetime   -= dt
            storm.intensity  -= 0.005 * dt   # slow decay

        # Remove dead storms
        self._storms = [s for s in self._storms if s.lifetime > 0.0 and s.intensity > 0.01]

        # Periodically check for new storm birth
        self._storm_check_counter += 1
        if self._storm_check_counter >= self._storm_check_interval:
            self._storm_check_counter = 0
            self._try_spawn_storms()

    def _try_spawn_storms(self) -> None:
        """Stochastically spawn storm cells where conditions are met."""
        max_storms = 5
        candidates_to_check = min(20, self._n)
        for _ in range(candidates_to_check):
            idx = self._rng.randint(0, self._n - 1)
            y   = idx // self._w
            x   = idx %  self._w
            lat = self._row_lats[y]
            lon = self._col_lons[x]
            u   = self._wind_u[idx]
            v   = self._wind_v[idx]
            wspd = math.sqrt(u * u + v * v)

            if (self._dust[idx] < self._storm_dust_threshold or
                    wspd < self._storm_wind_threshold or
                    len(self._storms) >= max_storms):
                continue

            # Spawn probability proportional to excess dust and wind
            dust_excess = self._dust[idx] - self._storm_dust_threshold
            wind_excess = wspd - self._storm_wind_threshold
            prob = dust_excess * wind_excess / (
                (1.0 - self._storm_dust_threshold) * self._storm_wind_threshold * 10.0
            )
            if self._rng.random() < _clamp(prob, 0.0, 0.5):
                self._storms.append(StormCell(
                    center_lat=lat,
                    center_lon=lon,
                    radius=0.3,           # ~17 degrees angular
                    intensity=0.5 + 0.3 * self._rng.random(),
                    vel_u=u,
                    vel_v=v,
                    lifetime=300.0,       # 5 minutes game-time
                ))

    # ------------------------------------------------------------------
    # Private grid / sampling helpers
    # ------------------------------------------------------------------

    def _wind_base_u(self, lat_rad: float) -> float:
        """Idealised zonal base wind by latitude (three-cell circulation)."""
        frac = abs(lat_rad) / _HALF_PI   # 0 = equator, 1 = pole
        if frac < 0.30:
            # Equatorial trade winds: westward (negative u)
            return -self._wind_base_strength * (1.0 - frac / 0.30)
        elif frac < 0.70:
            # Mid-latitude westerlies: eastward (positive u)
            t = (frac - 0.30) / 0.40
            return self._wind_base_strength * t
        else:
            # Polar easterlies: weak westward
            return -self._wind_base_strength * 0.5

    def _lat_lon_to_grid(self, lat_rad: float, lon_rad: float) -> tuple[float, float]:
        """Convert lat/lon (radians) to fractional grid coordinates."""
        fx = (lon_rad + math.pi) / _TWO_PI  * self._w - 0.5
        fy = (lat_rad + _HALF_PI) / math.pi * self._h - 0.5
        return fx, fy

    def _bilerp(self, field: list[float], fx: float, fy: float) -> float:
        """Bilinear interpolation from a flat W×H field.

        Longitude (x) wraps; latitude (y) clamps at poles.
        """
        W, H = self._w, self._h
        # Wrap x
        fx = fx % W
        if fx < 0.0:
            fx += W
        # Clamp y
        fy = _clamp(fy, 0.0, H - 1.0 - 1e-9)

        x0 = int(fx);  x1 = (x0 + 1) % W
        y0 = int(fy);  y1 = min(y0 + 1, H - 1)
        tx = fx - x0
        ty = fy - y0

        v00 = field[y0 * W + x0]
        v10 = field[y0 * W + x1]
        v01 = field[y1 * W + x0]
        v11 = field[y1 * W + x1]
        return (v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty) + \
               (v01 * (1.0 - tx) + v11 * tx) * ty


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _angular_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle angular distance (radians) between two lat/lon points."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat * 0.5) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon * 0.5) ** 2
    return 2.0 * math.asin(math.sqrt(_clamp(a, 0.0, 1.0)))
