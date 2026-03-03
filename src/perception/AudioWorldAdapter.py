"""AudioWorldAdapter — Stage 46 bridge from AcousticWorld to PerceptionSystem.

Translates the outputs of the Stage 46 acoustic world model
(:class:`~audio_world.EmitterAggregator`,
:class:`~audio_world.PropagationModel`,
:class:`~audio_world.InfraField`) into the
:class:`~perception.AudioSalience.AudioSource` list and infrasound fields
that :class:`~perception.PerceptionSystem.PerceptionSystem` consumes.

Also fills the extended perception fields introduced in Stage 46:
* ``audioSNR``      — signal-to-noise ratio proxy [0..1]
* ``infraUrgency``  — infrasound urgency [0..1]

Public API
----------
AudioWorldAdapter(config=None)
  .build_audio_sources(listener_pos, emitters, propagation_fn,
                       infra_field, dust_density, wind_speed)
    → AudioWorldResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.math.Vec3 import Vec3
from src.perception.AudioSalience import AudioSource
from src.audio.audio_world.EmitterAggregator import AcousticEmitterRecord
from src.audio.audio_world.PropagationModel import PropagationModel, PropagationResult
from src.audio.audio_world.InfraField import InfraField


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class AudioWorldResult:
    """Outputs of the adapter for one tick.

    Attributes
    ----------
    audio_sources :
        List of :class:`AudioSource` objects ready for PerceptionSystem.
    audio_snr :
        Worst-case (lowest) SNR across all active emitters [0..1].
    infra_urgency :
        Infrasound urgency at listener position [0..1].
    vibration_level :
        Combined vibration level at listener position [0..1].
    """
    audio_sources:   List[AudioSource] = field(default_factory=list)
    audio_snr:       float             = 1.0
    infra_urgency:   float             = 0.0
    vibration_level: float             = 0.0


class AudioWorldAdapter:
    """Convert AcousticWorld state into PerceptionSystem inputs.

    Parameters
    ----------
    config :
        Optional dict; reads ``audio_world.*`` keys.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._prop_model = PropagationModel(config)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build_audio_sources(
        self,
        listener_pos:  Vec3,
        emitters:      List[AcousticEmitterRecord],
        infra_field:   Optional[InfraField] = None,
        dust_density:  float = 0.0,
        wind_speed:    float = 0.0,
        occlusion_fn:  Optional[Callable[[int], float]] = None,
        valley_fn:     Optional[Callable[[Vec3], float]] = None,
    ) -> AudioWorldResult:
        """Translate emitter records into perception-ready audio sources.

        Parameters
        ----------
        listener_pos :
            Character world position.
        emitters :
            Active :class:`AcousticEmitterRecord` list from
            :class:`~audio_world.EmitterAggregator`.
        infra_field :
            Optional :class:`InfraField`; queried at *listener_pos*.
        dust_density :
            Atmospheric dust [0..1].
        wind_speed :
            Normalised wind speed [0..1].
        occlusion_fn :
            Optional ``(emitter_id) -> float`` returning cached occlusion
            [0..1].  Defaults to 0 (no occlusion).
        valley_fn :
            Optional ``(pos: Vec3) -> float`` returning valley gain at
            *pos*.  Defaults to 1.0.
        """
        sources: List[AudioSource] = []
        min_snr = 1.0

        for emitter in emitters:
            diff = emitter.pos - listener_pos
            dist = diff.length()
            if dist < 1e-6:
                dist = 1e-6

            occlusion  = occlusion_fn(emitter.id) if occlusion_fn else 0.0
            valley_gain = valley_fn(listener_pos) if valley_fn else 1.0

            result: PropagationResult = self._prop_model.propagate(
                distance=dist,
                band_energy_audible=emitter.band_energy_audible,
                band_energy_infra=emitter.band_energy_infra,
                dust_density=dust_density,
                wind_speed=wind_speed,
                occlusion=occlusion,
                valley_gain=valley_gain,
            )

            effective_energy = result.gain_audible
            if effective_energy < 1e-6:
                continue

            # low_freq_ratio: infra portion of total perceived energy
            total_gain = result.gain_audible + result.gain_infra
            lf_ratio = (result.gain_infra / total_gain) if total_gain > 1e-9 else 0.0

            sources.append(AudioSource(
                position=emitter.pos,
                energy=_clamp(effective_energy, 0.0, 1.0),
                low_freq_ratio=_clamp(lf_ratio, 0.0, 1.0),
            ))

            if result.snr < min_snr:
                min_snr = result.snr

        # Infra field at listener
        infra_urgency = 0.0
        if infra_field is not None:
            infra_urgency = infra_field.urgency(listener_pos)

        vibration_level = _clamp(infra_urgency + (1.0 - min_snr) * 0.2, 0.0, 1.0)

        return AudioWorldResult(
            audio_sources=sources,
            audio_snr=min_snr,
            infra_urgency=infra_urgency,
            vibration_level=vibration_level,
        )
