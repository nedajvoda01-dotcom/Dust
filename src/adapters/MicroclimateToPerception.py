"""MicroclimateToPerception — Stage 49 microclimate → Perception (37) adapter.

Translates :class:`~src.microclimate.LocalClimateComposer.LocalClimate`
into the inputs expected by :class:`~src.perception.WindLoad.WindLoadField`
and :class:`~src.perception.VisibilityEstimator.VisibilityEstimator`.

* ``windLoad`` is derived from local wind speed (post-shelter/channeling).
* ``shelter_factor`` is passed directly from LocalClimate.shelter.
* ``dust_density`` is the locally-adjusted dust concentration.

Public API
----------
MicroclimateToPerception()
  .wind_inputs(local_climate, macro_wind_vec) → dict
  .visibility_inputs(local_climate) → dict
"""
from __future__ import annotations

from src.math.Vec3 import Vec3
from src.microclimate.LocalClimateComposer import LocalClimate


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class MicroclimateToPerception:
    """Bridges LocalClimate to WindLoadField and VisibilityEstimator inputs."""

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def wind_inputs(
        self,
        local_climate:  LocalClimate,
        macro_wind_vec: Vec3,
        full_load_speed: float = 25.0,
    ) -> dict:
        """Build kwargs for :meth:`~src.perception.WindLoad.WindLoadField.update`.

        Parameters
        ----------
        local_climate :
            Locally-adjusted climate.
        macro_wind_vec :
            The macro wind velocity vector (world space, m/s).
        full_load_speed :
            Wind speed [m/s] that maps to normalised speed 1.0.

        Returns
        -------
        dict
            Keys: ``wind_vec``, ``shelter_factor``, ``dust_wall_near``.
        """
        # Scale macro wind vector by the local speed ratio
        macro_speed = macro_wind_vec.length()
        if macro_speed > 1e-6:
            scale = local_climate.wind_speed / max(
                local_climate.wind_speed + 1e-6, macro_speed / full_load_speed
            )
            local_vec = Vec3(
                macro_wind_vec.x * local_climate.wind_speed / (macro_speed / full_load_speed + 1e-6),
                macro_wind_vec.y * local_climate.wind_speed / (macro_speed / full_load_speed + 1e-6),
                macro_wind_vec.z * local_climate.wind_speed / (macro_speed / full_load_speed + 1e-6),
            )
        else:
            local_vec = Vec3(0.0, 0.0, 0.0)

        return {
            "wind_vec":       local_vec,
            "shelter_factor": _clamp(local_climate.shelter),
            "dust_wall_near": _clamp(local_climate.dust_density),
        }

    def visibility_inputs(self, local_climate: LocalClimate) -> dict:
        """Build kwargs for visibility estimator update.

        Returns
        -------
        dict
            Keys: ``dust_density``.
        """
        return {"dust_density": _clamp(local_climate.dust_density)}
