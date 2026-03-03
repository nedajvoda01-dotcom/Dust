"""MicroclimateToMaterials — Stage 49 microclimate → Materials (45) adapter.

Translates :class:`~src.microclimate.LocalClimateComposer.LocalClimate`
into a :class:`~src.material.PhaseChangeSystem.ClimateSample` that
the PhaseChangeSystem can consume.

Key effects
-----------
* **iceFilm** forms faster when ``cold_bias`` is high and ``temp_proxy`` low.
* **Dust deposition** increases when ``dust_trap`` is high.
* **Roughness polishing** increases when ``wind_channel`` is high.

Public API
----------
MicroclimateToMaterials()
  .to_climate_sample(local_climate, macro_sample) → ClimateSample
"""
from __future__ import annotations

from src.material.PhaseChangeSystem import ClimateSample
from src.microclimate.LocalClimateComposer import LocalClimate


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class MicroclimateToMaterials:
    """Bridges LocalClimate to :class:`~src.material.PhaseChangeSystem.ClimateSample`.

    Parameters
    ----------
    cold_bias_temp_penalty :
        How much each unit of cold_bias reduces the temperature proxy.
        Default 0.3 matches ``micro.coldbias.cold_delta``.
    """

    def __init__(self, cold_bias_temp_penalty: float = 0.3) -> None:
        self._cold_penalty = cold_bias_temp_penalty

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def to_climate_sample(
        self,
        local_climate: LocalClimate,
        macro_sample:  ClimateSample,
    ) -> ClimateSample:
        """Produce a locally-adjusted ClimateSample.

        Parameters
        ----------
        local_climate :
            Locally-adjusted climate from LocalClimateComposer.
        macro_sample :
            The original macro ClimateSample (provides storm flag, moisture).

        Returns
        -------
        ClimateSample
            With locally-adjusted wind_speed, dust_density, temperature, and
            insolation.
        """
        # Temperature: LocalClimate.temp_proxy already incorporates cold_bias
        adj_temp = _clamp(local_climate.temp_proxy)

        # Insolation: reduce if cold_bias is high (shadowed/enclosed)
        adj_insolation = _clamp(macro_sample.insolation * (1.0 - local_climate.cold_bias * 0.5))

        return ClimateSample(
            wind_speed=_clamp(local_climate.wind_speed),
            dust_density=_clamp(local_climate.dust_density),
            insolation=adj_insolation,
            temperature=adj_temp,
            moisture=macro_sample.moisture,
            storm_active=macro_sample.storm_active,
        )
