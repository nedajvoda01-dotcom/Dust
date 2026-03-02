"""ClimateCouplerAdapter — Stage 45 climate → ClimateSample bridge.

Extracts the per-cell drivers required by :class:`PhaseChangeSystem`
from the existing climate and astro-coupler interfaces (Stages 29/32).

The adapter is intentionally thin: it calls into ClimateSystem and
AstroClimateCoupler using their existing public APIs and packages the
result into a :class:`~src.material.PhaseChangeSystem.ClimateSample`.

Public API
----------
ClimateCouplerAdapter(climate_system, astro_coupler=None)
  .sample(world_pos) -> ClimateSample
"""
from __future__ import annotations

from src.material.PhaseChangeSystem import ClimateSample


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


class ClimateCouplerAdapter:
    """Bridges existing climate/astro systems to :class:`ClimateSample`.

    Parameters
    ----------
    climate_system :
        A :class:`~src.systems.ClimateSystem.ClimateSystem` (or stub)
        instance that implements ``sample_wind``, ``sample_dust``,
        ``sample_temperature``, ``get_wetness``.
    astro_coupler : optional
        An :class:`~src.systems.AstroClimateCoupler.AstroClimateCoupler`
        instance for ice-formation rates and insolation.  If None,
        insolation defaults to 0.5.
    mega_event_system : optional
        Provides ``is_storm_active()`` for storm flag.
    """

    def __init__(
        self,
        climate_system,
        astro_coupler=None,
        mega_event_system=None,
    ) -> None:
        self._climate = climate_system
        self._astro   = astro_coupler
        self._mega    = mega_event_system

    def sample(self, world_pos) -> ClimateSample:
        """Build a :class:`ClimateSample` for *world_pos*.

        Parameters
        ----------
        world_pos :
            A ``Vec3`` (or any object accepted by the underlying systems).
        """
        # Wind speed (magnitude normalised 0..1, assuming max ~30 m/s)
        wind_vec = self._climate.sample_wind(world_pos)
        wind_speed = _clamp(wind_vec.length() / 30.0 if hasattr(wind_vec, "length")
                            else (wind_vec[0] ** 2 + wind_vec[1] ** 2 + wind_vec[2] ** 2) ** 0.5 / 30.0)

        dust = _clamp(self._climate.sample_dust(world_pos))

        # Temperature: ClimateSystem returns a raw value; normalise to [0,1]
        # assuming range 0..400 K  (arbitrary planet range)
        raw_temp = self._climate.sample_temperature(world_pos)
        temperature = _clamp(raw_temp / 400.0 if raw_temp > 1.0 else raw_temp)

        moisture = 0.0
        if hasattr(self._climate, "get_wetness"):
            moisture = _clamp(self._climate.get_wetness(world_pos))

        # Insolation from astro coupler (ice_form_rate is inverse proxy)
        insolation = 0.5
        if self._astro is not None:
            ice_rate = _clamp(self._astro.ice_form_rate(world_pos))
            # High ice formation → low insolation
            insolation = _clamp(1.0 - ice_rate)

        # Storm flag from mega event system
        storm_active = False
        if self._mega is not None and hasattr(self._mega, "is_storm_active"):
            storm_active = bool(self._mega.is_storm_active())

        return ClimateSample(
            wind_speed=wind_speed,
            dust_density=dust,
            insolation=insolation,
            temperature=temperature,
            moisture=moisture,
            storm_active=storm_active,
        )
