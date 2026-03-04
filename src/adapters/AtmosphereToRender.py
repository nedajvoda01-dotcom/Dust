"""AtmosphereToRender — Stage 64 adapter: atmosphere → render parameter bundle.

Translates LocalAtmoParams into render parameters consumed by Stage 62
(visual identity) and Stage 65 (volumetric fields).

All outputs are normalised floats — no textures, only fields and curves.

Output parameters
-----------------
fog_density          float [0..1]  — volumetric fog density
scattering_strength  float [0..1]  — in-scattering intensity
dust_color_shift     float [0..1]  — hue/saturation shift from aerosol tint
sun_shafts_intensity float [0..1]  — god-ray / crepuscular intensity
ring_shadow_mod      float [0..1]  — ring-shadow modulation (from pressure)
visibility_scale     float [0..1]  — overall visibility multiplier

Public API
----------
AtmosphereToRender(config=None)
  .render_params(local_params) -> AtmoRenderParams
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.atmo.AtmosphereSystem       import LocalAtmoParams
from src.atmo.WeatherRegimeDetector  import WeatherRegime


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class AtmoRenderParams:
    fog_density:          float = 0.0
    scattering_strength:  float = 0.3
    dust_color_shift:     float = 0.0
    sun_shafts_intensity: float = 0.0
    ring_shadow_mod:      float = 0.5
    visibility_scale:     float = 1.0


class AtmosphereToRender:
    """Map LocalAtmoParams to render modulation parameters.

    Parameters
    ----------
    config :
        Optional dict; reads ``atmo64.render.*`` keys.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        pass   # no tunable params in v1; reserved for future config

    def render_params(self, local_params: LocalAtmoParams) -> AtmoRenderParams:
        """Compute render parameters from *local_params*."""
        regime = local_params.regime
        fp     = local_params.fog_potential
        ae     = local_params.aerosol
        sp     = local_params.storm_potential
        vis    = local_params.visibility

        # Fog density from fog potential
        fog = _clamp(fp * 1.2) if regime == WeatherRegime.FOG else _clamp(fp * 0.4)

        # Scattering: higher in dust storm (forward scattering) and fog
        scatter = _clamp(0.3 + ae * 0.5 + fp * 0.3)

        # Dust color shift (orange tint) scales with aerosol density
        color_shift = _clamp(ae * 0.8)

        # Sun shafts brighten in low-fog, moderate-dust conditions
        shafts = _clamp((1.0 - fog) * ae * 0.6)

        # Ring shadow modulation (proxy via pressure; low P → deeper shadow)
        ring_mod = _clamp(0.3 + local_params.pressure * 0.4)

        return AtmoRenderParams(
            fog_density          = fog,
            scattering_strength  = scatter,
            dust_color_shift     = color_shift,
            sun_shafts_intensity = shafts,
            ring_shadow_mod      = ring_mod,
            visibility_scale     = vis,
        )
