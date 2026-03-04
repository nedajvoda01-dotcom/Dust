"""FieldAdvection — Stage 64 simplified advection and relaxation update.

Implements one tick of the atmosphere dynamics update:
  1. Advect A (aerosol) and H (humidity) along the wind field.
  2. Relax T toward an equilibrium (insolation-driven).
  3. Derive P from T gradients + background map.
  4. Recompute wind from P gradients + base circulation.

All operations are on a :class:`~src.atmo.GlobalFieldGrid.GlobalFieldGrid`
and are applied **in-place** (new tile objects are written back).

Public API
----------
FieldAdvection(config=None)
  .step(grid, dt, insolation_map=None) -> None
      Advance all fields by *dt* seconds.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional

from src.atmo.GlobalFieldGrid import GlobalFieldGrid, AtmoTile


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class FieldAdvection:
    """Simplified advection + relaxation step for atmospheric fields.

    Parameters
    ----------
    config :
        Optional dict; reads ``atmo64.*`` keys.
    """

    # Configurable defaults (AI-adjustable within allowlist)
    _DEFAULT_TEMP_RELAX_TAU     = 300.0   # s — temperature equilibrium time
    _DEFAULT_PRESSURE_RELAX_TAU = 150.0   # s — pressure equilibrium time
    _DEFAULT_AEROSOL_SETTLE     = 0.002   # per-second gravitational settling
    _DEFAULT_EROSION_LIFT       = 0.001   # per-second wind-driven lift rate
    _DEFAULT_WIND_ADVECT_SCALE  = 0.3     # fraction of wind used for advection
    _DEFAULT_BASE_CIRCULATION   = 0.02    # background planetary circulation

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg  = config or {}
        acfg = cfg.get("atmo64", {}) or {}

        self._temp_tau:    float = float(acfg.get("temp_relax_tau",     self._DEFAULT_TEMP_RELAX_TAU))
        self._p_tau:       float = float(acfg.get("pressure_relax_tau", self._DEFAULT_PRESSURE_RELAX_TAU))
        self._settle:      float = float(acfg.get("aerosol_settle_rate", self._DEFAULT_AEROSOL_SETTLE))
        self._lift:        float = float(acfg.get("erosion_lift_rate",   self._DEFAULT_EROSION_LIFT))
        self._adv_scale:   float = float(acfg.get("wind_advect_scale",   self._DEFAULT_WIND_ADVECT_SCALE))
        self._base_circ:   float = float(acfg.get("base_circulation",    self._DEFAULT_BASE_CIRCULATION))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def step(
        self,
        grid: GlobalFieldGrid,
        dt: float,
        insolation_map: Optional[Callable[[int, int], float]] = None,
    ) -> None:
        """Advance all atmospheric fields by *dt* seconds.

        Parameters
        ----------
        grid :
            The field grid to update in-place.
        dt :
            Time step in seconds.
        insolation_map :
            Optional callable ``(ix, iy) -> float [0..1]`` returning the
            fraction of stellar insolation at each tile.  Defaults to 0.5
            (neutral) everywhere.
        """
        w, h = grid.width, grid.height
        old_tiles = [grid.tile(ix, iy) for iy in range(h) for ix in range(w)]

        def _old(ix: int, iy: int) -> AtmoTile:
            return old_tiles[iy * w + ix]

        insol = insolation_map or (lambda ix, iy: 0.5)

        # Alpha factors for relaxation
        alpha_T = 1.0 - math.exp(-dt / max(self._temp_tau, 1e-3))
        alpha_P = 1.0 - math.exp(-dt / max(self._p_tau,    1e-3))

        for iy in range(h):
            for ix in range(w):
                t = _old(ix, iy)
                sol = _clamp(insol(ix, iy))

                # ---- Temperature relaxation toward insolation equilibrium ----
                new_T = t.temperature + alpha_T * (sol - t.temperature)

                # ---- Pressure: background + gradient from T ----------------
                # Background pressure: 0.5 + offset from T (warm → higher P)
                p_eq = _clamp(0.4 + 0.2 * new_T)
                # Local correction from neighbouring T gradients (proxy for
                # warm-core high / cold-core low)
                tl = _old(max(ix - 1, 0),     iy).temperature
                tr = _old(min(ix + 1, w - 1), iy).temperature
                tu = _old(ix, max(iy - 1, 0)).temperature
                td = _old(ix, min(iy + 1, h - 1)).temperature
                t_laplacian = (tl + tr + tu + td - 4.0 * t.temperature)
                p_eq = _clamp(p_eq + 0.05 * t_laplacian)
                new_P = t.pressure + alpha_P * (p_eq - t.pressure)

                # ---- Wind from pressure gradient + base circulation --------
                pl = _old(max(ix - 1, 0),     iy).pressure
                pr = _old(min(ix + 1, w - 1), iy).pressure
                pu = _old(ix, max(iy - 1, 0)).pressure
                pd = _old(ix, min(iy + 1, h - 1)).pressure
                # Wind blows from high P to low P (negative gradient)
                p_grad_x = -(pr - pl) * 0.5
                p_grad_y = -(pd - pu) * 0.5
                # Add simple Hadley-cell-like base circulation (zonal)
                base_x = self._base_circ * math.cos(iy / max(h - 1, 1) * math.pi)
                new_Vx = _clamp(p_grad_x + base_x, -1.0, 1.0)
                new_Vy = _clamp(p_grad_y, -1.0, 1.0)

                # ---- Aerosol advection + settle + erosion lift ----------------
                # Advect from upstream tile (upwind scheme)
                adv_x = t.wind_x * self._adv_scale
                adv_y = t.wind_y * self._adv_scale
                src_ix = int(round(_clamp(ix - adv_x * dt, 0, w - 1)))
                src_iy = int(round(_clamp(iy - adv_y * dt, 0, h - 1)))
                src_A  = _old(src_ix, src_iy).aerosol
                new_A  = _clamp(
                    src_A
                    - self._settle * dt * src_A
                    + self._lift   * dt * t.wind_speed * (1.0 - src_A)
                )

                # ---- Humidity advection + condensation/sublimation (proxy) ---
                src_H  = _old(src_ix, src_iy).humidity
                # Humidity condenses when cold and high P
                condense = _clamp(
                    (1.0 - new_T) * new_P * 0.01 * dt * src_H
                )
                new_H = _clamp(src_H - condense)

                # ---- Electro activity (proportional to storm potential) ------
                ws = math.sqrt(new_Vx ** 2 + new_Vy ** 2)
                p_grad_mag = math.sqrt(p_grad_x ** 2 + p_grad_y ** 2)
                storm = _clamp(p_grad_mag * 0.4 + ws * 0.4 + new_A * 0.2)
                # Relax electro toward storm potential
                new_E = _clamp(t.electro + 0.1 * (storm - t.electro) * dt)

                grid.set_tile(ix, iy, AtmoTile(
                    pressure    = new_P,
                    temperature = new_T,
                    wind_x      = new_Vx,
                    wind_y      = new_Vy,
                    aerosol     = new_A,
                    humidity    = new_H,
                    electro     = new_E,
                ))
