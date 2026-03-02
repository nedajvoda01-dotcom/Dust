"""PerceptionSystem — Stage 37 Perception Field orchestrator.

"Nervous system" for the character: aggregates environmental fields into
the :class:`PerceptionState` struct that MotorStack reads every tick.

Architecture
------------
::

    AudioSalienceField    ─┐
    GroundStabilityField  ─┤
    WindLoadField         ─┼─► ThreatAggregator ─► PerceptionState
    VisibilityEstimator   ─┤
    VibrationField        ─┤
    PresenceField         ─┘

Each sub-field has its own update rate (``tick_hz``).  The system caches
outputs between ticks and only recomputes when a sub-field's timer fires.

Public API
----------
PerceptionSystem(config=None, sim_time=0.0)
  .update(dt, sim_time, env) → PerceptionState
  .debug_info()              → dict

``env`` is a :class:`PerceptionEnv` dataclass — caller fills in whatever
is available; missing fields fall back to neutral defaults.

PerceptionState
---------------
All outputs are [0..1] scalars or Vec3 unit vectors.

  globalRisk          float
  attentionDir        Vec3
  movementConfidence  float
  braceBias           float
  slipRisk            float
  windLoad            float
  visibility          float
  presenceNear        float
  assistOpportunity   float
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from src.math.Vec3 import Vec3
from src.perception.AudioSalience      import AudioSalienceField, AudioSource
from src.perception.GroundStability    import GroundStabilityField
from src.perception.WindLoad           import WindLoadField
from src.perception.VisibilityEstimator import VisibilityEstimator
from src.perception.VibrationField     import VibrationField, GeoVibrationSignal
from src.perception.PresenceField      import PresenceField, OtherPlayerState
from src.perception.ThreatAggregator   import ThreatAggregator


# ---------------------------------------------------------------------------
# PerceptionEnv — caller-provided environmental snapshot
# ---------------------------------------------------------------------------

@dataclass
class PerceptionEnv:
    """All environmental inputs for one perception update.

    Fields are optional; unset fields use neutral defaults.
    """
    # --- Audio
    audio_sources:  List[AudioSource]       = field(default_factory=list)
    dust_density:   float                   = 0.0
    cave_factor:    float                   = 0.0

    # --- Ground
    friction:       float                   = 1.0
    softness:       float                   = 0.0
    slope_deg:      float                   = 0.0
    roughness:      float                   = 0.5

    # --- Wind
    wind_vec:       Optional[Vec3]          = None
    shelter_factor: float                   = 0.0
    dust_wall_near: float                   = 0.0

    # --- Visibility
    fog:            float                   = 0.0
    night_factor:   float                   = 0.0

    # --- Vibration
    geo_signals:    List[GeoVibrationSignal] = field(default_factory=list)
    bulk_lf_energy: float                    = 0.0

    # --- Presence
    others:         List[OtherPlayerState]  = field(default_factory=list)

    # --- Self position
    position:       Optional[Vec3]          = None


# ---------------------------------------------------------------------------
# PerceptionState — output struct consumed by MotorStack
# ---------------------------------------------------------------------------

@dataclass
class PerceptionState:
    """Aggregated perception outputs for MotorStack.

    All scalars are in [0..1]; directions are unit Vec3.
    """
    globalRisk:         float = 0.0
    attentionDir:       Vec3  = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    movementConfidence: float = 1.0
    braceBias:          float = 0.0
    slipRisk:           float = 0.0
    windLoad:           float = 0.0
    visibility:         float = 1.0
    presenceNear:       float = 0.0
    assistOpportunity:  float = 0.0


# ---------------------------------------------------------------------------
# PerceptionSystem
# ---------------------------------------------------------------------------

class PerceptionSystem:
    """Orchestrates all perception sub-fields and emits PerceptionState.

    Parameters
    ----------
    config :
        Optional dict; reads ``perception.*`` keys.
    sim_time :
        Initial simulation time [s].
    """

    _DEFAULT_TICK_HZ          = 20.0
    _DEFAULT_AUDIO_HZ         = 20.0
    _DEFAULT_GROUND_HZ        = 10.0
    _DEFAULT_WIND_HZ          = 10.0
    _DEFAULT_VIS_HZ           =  5.0
    _DEFAULT_VIBRATION_HZ     = 10.0
    _DEFAULT_PRESENCE_HZ      = 20.0

    def __init__(
        self,
        config:   Optional[dict] = None,
        sim_time: float           = 0.0,
    ) -> None:
        self._cfg = config or {}
        pcfg = self._cfg.get("perception", {}) or {}

        # Sub-field update intervals
        def _hz(key: str, default: float) -> float:
            return 1.0 / max(1.0, float(pcfg.get(key, default)))

        self._dt_audio    = _hz("audio_hz",    self._DEFAULT_AUDIO_HZ)
        self._dt_ground   = _hz("ground_hz",   self._DEFAULT_GROUND_HZ)
        self._dt_wind     = _hz("wind_hz",     self._DEFAULT_WIND_HZ)
        self._dt_vis      = _hz("vis_hz",      self._DEFAULT_VIS_HZ)
        self._dt_vibr     = _hz("vibration_hz",self._DEFAULT_VIBRATION_HZ)
        self._dt_presence = _hz("presence_hz", self._DEFAULT_PRESENCE_HZ)

        # Timers (next scheduled tick time)
        self._t_audio    = sim_time
        self._t_ground   = sim_time
        self._t_wind     = sim_time
        self._t_vis      = sim_time
        self._t_vibr     = sim_time
        self._t_presence = sim_time

        # Sub-fields
        self._audio    = AudioSalienceField(config)
        self._ground   = GroundStabilityField(config)
        self._wind     = WindLoadField(config)
        self._vis      = VisibilityEstimator(config)
        self._vibr     = VibrationField(config)
        self._presence = PresenceField(config)
        self._threat   = ThreatAggregator()

        # Cached PerceptionState
        self._state = PerceptionState()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        dt:       float,
        sim_time: float,
        env:      PerceptionEnv,
    ) -> PerceptionState:
        """Advance the perception system and return updated PerceptionState.

        Parameters
        ----------
        dt :
            Elapsed wall/sim time since last call [s].
        sim_time :
            Current absolute simulation time [s].
        env :
            Environmental snapshot provided by the caller.
        """
        pos = env.position or Vec3(0.0, 0.0, 0.0)

        # Tick each sub-field at its own rate
        if sim_time >= self._t_audio:
            self._audio.update(
                listener_pos=pos,
                sources=env.audio_sources,
                dust_density=env.dust_density,
                cave_factor=env.cave_factor,
                dt=self._dt_audio,
            )
            self._t_audio = sim_time + self._dt_audio

        if sim_time >= self._t_ground:
            self._ground.update(
                friction=env.friction,
                softness=env.softness,
                slope_deg=env.slope_deg,
                roughness=env.roughness,
                dt=self._dt_ground,
            )
            self._t_ground = sim_time + self._dt_ground

        if sim_time >= self._t_wind:
            self._wind.update(
                wind_vec=env.wind_vec,
                shelter_factor=env.shelter_factor,
                dust_wall_near=env.dust_wall_near,
                dt=self._dt_wind,
            )
            self._t_wind = sim_time + self._dt_wind

        if sim_time >= self._t_vis:
            self._vis.update(
                dust_density=env.dust_density,
                fog=env.fog,
                night_factor=env.night_factor,
                dt=self._dt_vis,
            )
            self._t_vis = sim_time + self._dt_vis

        if sim_time >= self._t_vibr:
            self._vibr.update(
                listener_pos=pos,
                geo_signals=env.geo_signals,
                bulk_lf_energy=env.bulk_lf_energy,
                dt=self._dt_vibr,
            )
            self._t_vibr = sim_time + self._dt_vibr

        if sim_time >= self._t_presence:
            self._presence.update(
                listener_pos=pos,
                others=env.others,
                self_global_risk=self._state.globalRisk,
                dt=self._dt_presence,
            )
            self._t_presence = sim_time + self._dt_presence

        # Aggregate
        self._threat.update(
            slip_risk=self._ground.slip_risk,
            sink_risk=self._ground.sink_risk,
            vibration_level=self._vibr.vibration_level,
            wind_load=self._wind.wind_load,
            visibility=self._vis.visibility,
            audio_salience=self._audio.audio_salience,
            audio_urgency=self._audio.audio_urgency,
            audio_dir=self._audio.audio_dir,
            vibration_dir=self._vibr.vibration_dir,
            presence_near=self._presence.presence_near,
            assist_opportunity=self._presence.assist_opportunity,
            presence_dir=self._presence.presence_dir,
        )

        self._state = PerceptionState(
            globalRisk=self._threat.global_risk,
            attentionDir=self._threat.attention_dir,
            movementConfidence=self._threat.movement_confidence,
            braceBias=self._threat.brace_bias,
            slipRisk=self._ground.slip_risk,
            windLoad=self._wind.wind_load,
            visibility=self._vis.visibility,
            presenceNear=self._presence.presence_near,
            assistOpportunity=self._presence.assist_opportunity,
        )
        return self._state

    def debug_info(self) -> dict:
        """Return a dict of current sub-field values for logging / gizmos."""
        s = self._state
        return {
            "globalRisk":         s.globalRisk,
            "attentionDir":       (s.attentionDir.x, s.attentionDir.y, s.attentionDir.z),
            "movementConfidence": s.movementConfidence,
            "braceBias":          s.braceBias,
            "slipRisk":           s.slipRisk,
            "windLoad":           s.windLoad,
            "visibility":         s.visibility,
            "presenceNear":       s.presenceNear,
            "assistOpportunity":  s.assistOpportunity,
            "audioSalience":      self._audio.audio_salience,
            "audioUrgency":       self._audio.audio_urgency,
            "vibrationLevel":     self._vibr.vibration_level,
            "contrast":           self._vis.contrast,
            "supportQuality":     self._ground.support_quality,
            "gustiness":          self._wind.gustiness,
        }
