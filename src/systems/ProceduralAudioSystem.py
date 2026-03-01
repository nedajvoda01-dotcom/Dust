"""ProceduralAudioSystem — Stage 19 procedural sound synthesis without assets.

Generates and mixes all game audio entirely from DSP primitives (noise
generators, biquad filters, ADSR envelopes).  No external wav/ogg files are
loaded.

Architecture
------------
NoiseGenerator      — white / pink / brown noise (seeded, deterministic)
ToneGenerator       — sine / triangle low-frequency tones
BiquadFilter        — LP / HP / BP biquad (single-pole, sample-by-sample)
ADSR                — Attack-Decay-Sustain-Release amplitude envelope
AudioBurst          — a finite noise burst: envelope × noise × optional filter
AudioChannel        — named channel with its own synthesis state and gain
AudioMixer          — all channels, master limiter, Storm→Footsteps/Suit ducking
EventToAudioRouter  — subscribes to AnimEvent + GeoEventSignal and routes them
ProceduralAudioSystem — public API: update() / trigger_foot_plant() /
                        trigger_geo_event() / get_rms_levels()

All procedural variations are seeded deterministically:
  seed = global_seed XOR hash(event_id) XOR hash(time_bucket) XOR hash(material)

Channels
--------
WIND        ← windSpeed, gustStrength, stormIntensity, camera direction
STORM       ← dust, visibility, stormIntensity; ducts FOOTSTEPS + SUIT
FOOTSTEPS   ← OnFootPlant events with material + intensity
SUIT        ← effort, cadence, character state (breath, fabric, servos)
GEO_RUMBLE  ← GeoEventSignal PRE phase (infrasound, low-frequency drone)
GEO_IMPACT  ← GeoEventSignal IMPACT phase (thump + debris cascade)
UI          ← reserved, always silent

Public API
----------
ProceduralAudioSystem(config=None, global_seed=42)
  .update(dt, wind_speed, gust_strength, storm_intensity, dust, visibility,
          effort, cadence, character_state, anim_events, geo_signals,
          player_pos)  → None
  .trigger_foot_plant(foot, intensity, material_id, slide_vel)  → None
  .trigger_geo_signal(signal)                                    → None
  .get_rms_levels() → Dict[str, float]
  .get_debug_info() → dict

Material IDs (match ProceduralMaterialSystem rock_type_id)
-----------------------------------------------------------
0  DUST_LAYER
1  BASALT_ROCK
2  LOOSE_DEBRIS   (virtual — signalled by terrain context)
3  FRACTURED_ROCK
4  ICE_FILM
"""
from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from src.systems.GeoEventSystem import GeoEventPhase, GeoEventSignal
from src.systems.CharacterPhysicalController import CharacterState
from src.systems.ReflexSystem import AnimEvent, AnimEventType

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Material ID constants (mirror ProceduralMaterialSystem index)
# ---------------------------------------------------------------------------

MAT_DUST     = 0
MAT_ROCK     = 1
MAT_DEBRIS   = 2
MAT_FRACT    = 3
MAT_ICE      = 4

# ---------------------------------------------------------------------------
# Channel names
# ---------------------------------------------------------------------------

class AudioChannelName(Enum):
    WIND       = "Wind"
    STORM      = "Storm"
    FOOTSTEPS  = "Footsteps"
    SUIT       = "Suit"
    GEO_RUMBLE = "GeoRumble"
    GEO_IMPACT = "GeoImpact"
    UI         = "UI"


# ---------------------------------------------------------------------------
# Seeded RNG helpers
# ---------------------------------------------------------------------------

def _hash32(value: int) -> int:
    """Fast integer hash (non-cryptographic)."""
    x = value & 0xFFFFFFFF
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = x & 0xFFFFFFFF
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = x & 0xFFFFFFFF
    x = (x >> 16) ^ x
    return x


class _SeededRng:
    """Minimal LCG random number generator with explicit seed."""

    _A = 1664525
    _C = 1013904223
    _M = 2**32

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF

    def next_float(self) -> float:
        """Return next value in [0, 1)."""
        self._state = (self._A * self._state + self._C) % self._M
        return self._state / self._M

    def next_signed(self) -> float:
        """Return next value in (-1, 1)."""
        return self.next_float() * 2.0 - 1.0

    def fork(self, extra_seed: int) -> "_SeededRng":
        return _SeededRng(_hash32(self._state ^ extra_seed))


# ---------------------------------------------------------------------------
# Noise generators
# ---------------------------------------------------------------------------

class NoiseGenerator:
    """Sample-by-sample noise generation (white / pink / brown).

    All state is explicit so the generator can be driven at arbitrary rates
    without depending on a fixed sample rate.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = _SeededRng(seed)
        # Pink noise state (Voss-McCartney approximation, 7 octaves)
        self._pink_state: List[float] = [0.0] * 7
        self._pink_counter: int = 0
        # Brown noise integrator
        self._brown_prev: float = 0.0

    def white(self) -> float:
        """White noise sample in (-1, 1)."""
        return self._rng.next_signed()

    def pink(self) -> float:
        """Pink noise sample (1/f spectrum, Voss-McCartney approx)."""
        self._pink_counter = (self._pink_counter + 1) & 0xFFFFFF
        for i in range(7):
            if (self._pink_counter & (1 << i)) == 0:
                break
            self._pink_state[i] = self._rng.next_signed()
        raw = sum(self._pink_state) / 7.0
        return max(-1.0, min(1.0, raw))

    def brown(self) -> float:
        """Brown noise (1/f² — integrated white noise)."""
        step = self._rng.next_signed() * 0.1
        self._brown_prev = max(-1.0, min(1.0, self._brown_prev + step))
        return self._brown_prev


# ---------------------------------------------------------------------------
# Tone generator
# ---------------------------------------------------------------------------

class ToneGenerator:
    """Simple sine / triangle LFO / tone generator."""

    def __init__(self, freq_hz: float = 40.0, sample_rate: float = 60.0) -> None:
        self._phase = 0.0
        self._freq = freq_hz
        self._sr = sample_rate

    def set_freq(self, freq_hz: float) -> None:
        self._freq = freq_hz

    def sine(self) -> float:
        """Advance phase and return sine sample."""
        v = math.sin(self._phase * 2.0 * math.pi)
        self._phase = (self._phase + self._freq / self._sr) % 1.0
        return v

    def triangle(self) -> float:
        """Advance phase and return triangle wave sample."""
        p = (self._phase + self._freq / self._sr) % 1.0
        self._phase = p
        return 1.0 - abs(p * 4.0 - 2.0) if p < 0.5 else -(1.0 - abs((1.0 - p) * 4.0 - 2.0))


# ---------------------------------------------------------------------------
# Biquad filter (simplified, per-tick state)
# ---------------------------------------------------------------------------

class BiquadFilter:
    """Simple one-pole RC filter approximation used as LP / HP / BP.

    This is not a rigorous biquad but a fast RC approximation sufficient for
    real-time procedural audio characterisation (no actual audio output).
    """

    def __init__(self, cutoff_norm: float = 0.3, mode: str = "lp") -> None:
        """
        Parameters
        ----------
        cutoff_norm : float
            Normalised cutoff in (0, 1) relative to sample rate.
        mode : str
            "lp" (low-pass), "hp" (high-pass), "bp" (band-pass).
        """
        self._mode = mode
        self._alpha = cutoff_norm
        self._prev_lp: float = 0.0
        self._prev_hp: float = 0.0

    def set_cutoff(self, cutoff_norm: float) -> None:
        self._alpha = max(1e-4, min(0.9999, cutoff_norm))

    def process(self, x: float) -> float:
        lp = self._prev_lp + self._alpha * (x - self._prev_lp)
        hp = x - lp
        self._prev_lp = lp
        self._prev_hp = hp
        if self._mode == "lp":
            return lp
        if self._mode == "hp":
            return hp
        # band-pass: LP of HP
        bp = self._prev_hp + self._alpha * (hp - self._prev_hp)
        return bp


# ---------------------------------------------------------------------------
# ADSR envelope
# ---------------------------------------------------------------------------

class ADSR:
    """Simple ADSR envelope (level-based, dt-driven, no fixed sample rate)."""

    def __init__(
        self,
        attack:  float = 0.005,
        decay:   float = 0.05,
        sustain: float = 0.7,
        release: float = 0.1,
    ) -> None:
        self.attack  = attack
        self.decay   = decay
        self.sustain = sustain
        self.release = release
        self._level  = 0.0
        self._phase  = "idle"   # idle / attack / decay / sustain / release
        self._elapsed = 0.0

    def trigger(self) -> None:
        self._phase   = "attack"
        self._elapsed = 0.0

    def release_note(self) -> None:
        if self._phase not in ("idle", "release"):
            self._phase   = "release"
            self._elapsed = 0.0

    def advance(self, dt: float) -> float:
        """Advance envelope by *dt* seconds and return current amplitude."""
        self._elapsed += dt
        if self._phase == "idle":
            self._level = 0.0
        elif self._phase == "attack":
            self._level = min(1.0, self._elapsed / max(1e-9, self.attack))
            if self._elapsed >= self.attack:
                self._phase   = "decay"
                self._elapsed = 0.0
        elif self._phase == "decay":
            t = min(1.0, self._elapsed / max(1e-9, self.decay))
            self._level = 1.0 - t * (1.0 - self.sustain)
            if self._elapsed >= self.decay:
                self._phase   = "sustain"
        elif self._phase == "sustain":
            self._level = self.sustain
        elif self._phase == "release":
            start = self._level
            t = min(1.0, self._elapsed / max(1e-9, self.release))
            self._level = start * (1.0 - t)
            if self._elapsed >= self.release:
                self._phase = "idle"
                self._level = 0.0
        return self._level

    @property
    def is_idle(self) -> bool:
        return self._phase == "idle"


# ---------------------------------------------------------------------------
# AudioBurst — finite single event sound
# ---------------------------------------------------------------------------

@dataclass
class AudioBurst:
    """A finite sound event (footstep, geo-impact) rendered by AudioChannel."""
    rng:         _SeededRng
    duration:    float            # total seconds
    envelope:    ADSR
    filter:      Optional[BiquadFilter] = None
    noise_mode:  str              = "white"   # white / pink / brown
    gain:        float            = 1.0
    _elapsed:    float            = field(default=0.0, init=False)
    _noise_gen:  Optional[NoiseGenerator] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._noise_gen = NoiseGenerator(seed=self.rng._state)
        self.envelope.trigger()

    def tick(self, dt: float) -> float:
        """Return one sample value and advance state."""
        if self.is_done:
            return 0.0
        self._elapsed += dt
        amp = self.envelope.advance(dt)
        if self.noise_mode == "white":
            sample = self._noise_gen.white()
        elif self.noise_mode == "pink":
            sample = self._noise_gen.pink()
        else:
            sample = self._noise_gen.brown()
        if self.filter is not None:
            sample = self.filter.process(sample)
        return sample * amp * self.gain

    @property
    def is_done(self) -> bool:
        return self._elapsed >= self.duration


# ---------------------------------------------------------------------------
# AudioChannel
# ---------------------------------------------------------------------------

class AudioChannel:
    """One named audio channel with synthesis state, active bursts, RMS tracking."""

    def __init__(self, name: str, gain: float = 1.0) -> None:
        self.name          = name
        self.gain          = gain
        self.enabled       = True
        self._bursts:      List[AudioBurst] = []
        self._rms_window:  List[float] = [0.0] * 32   # short window
        self._rms_ptr:     int = 0
        self._last_sample: float = 0.0
        # Continuous synthesis state (for Wind, Storm, Suit, GeoRumble)
        self._cont_noise:  Optional[NoiseGenerator] = None
        self._cont_filter: Optional[BiquadFilter] = None
        self._cont_gain:   float = 0.0

    # --- burst management ---------------------------------------------------

    def add_burst(self, burst: AudioBurst) -> None:
        self._bursts.append(burst)

    # --- per-tick update ----------------------------------------------------

    def tick(self, dt: float, cont_sample: float = 0.0) -> float:
        """Advance channel one tick; return amplitude sample."""
        if not self.enabled:
            return 0.0
        sample = cont_sample * self._cont_gain
        alive: List[AudioBurst] = []
        for b in self._bursts:
            sample += b.tick(dt)
            if not b.is_done:
                alive.append(b)
        self._bursts = alive
        out = sample * self.gain
        self._rms_window[self._rms_ptr] = out * out
        self._rms_ptr = (self._rms_ptr + 1) % len(self._rms_window)
        self._last_sample = out
        return out

    @property
    def rms(self) -> float:
        mean_sq = sum(self._rms_window) / len(self._rms_window)
        return math.sqrt(max(0.0, mean_sq))

    @property
    def has_active_bursts(self) -> bool:
        return len(self._bursts) > 0


# ---------------------------------------------------------------------------
# AudioMixer
# ---------------------------------------------------------------------------

class AudioMixer:
    """Holds all channels, applies ducking, and runs a soft master limiter."""

    def __init__(self, config: dict) -> None:
        audio = config.get("audio", {})
        self.master_gain: float = float(audio.get("master_gain", 0.9))
        self._limiter_threshold: float = float(audio.get("limiter_threshold", 0.95))
        self._limiter_release:   float = float(audio.get("limiter_release", 0.2))
        self._duck_strength:     float = float(audio.get("storm_duck_strength", 0.7))
        self._limiter_gain:      float = 1.0

        self.channels: Dict[str, AudioChannel] = {}
        for ch in AudioChannelName:
            key  = ch.value
            gain_key = f"audio.{key.lower()}_gain"
            alt_key  = ch.name.lower() + "_gain"
            g = float(audio.get(alt_key, 1.0))
            self.channels[key] = AudioChannel(key, gain=g)

    # ------------------------------------------------------------------

    def tick(self, dt: float, cont_samples: Dict[str, float]) -> Dict[str, float]:
        """Advance all channels; apply ducking and limiter.

        Parameters
        ----------
        cont_samples : dict channel_name → continuous synthesis amplitude
        """
        # Evaluate Storm level first for ducking
        storm_ch   = self.channels[AudioChannelName.STORM.value]
        storm_rms  = storm_ch.rms
        duck_factor = max(0.0, 1.0 - storm_rms * self._duck_strength)

        raw: Dict[str, float] = {}
        for name, ch in self.channels.items():
            cs = cont_samples.get(name, 0.0)
            s  = ch.tick(dt, cs)
            # Apply ducking to Footsteps and Suit
            if name in (AudioChannelName.FOOTSTEPS.value, AudioChannelName.SUIT.value):
                s *= duck_factor
            raw[name] = s

        # Master limiter (soft-knee)
        master_mix = sum(abs(v) for v in raw.values())
        if master_mix > self._limiter_threshold:
            target = self._limiter_threshold / max(1e-9, master_mix)
            self._limiter_gain += (target - self._limiter_gain) * (1.0 - math.exp(-dt / max(1e-9, self._limiter_release)))
        else:
            self._limiter_gain = min(1.0, self._limiter_gain + dt / self._limiter_release)

        out = {k: v * self._limiter_gain * self.master_gain for k, v in raw.items()}
        return out

    def get_rms(self) -> Dict[str, float]:
        return {k: ch.rms for k, ch in self.channels.items()}


# ---------------------------------------------------------------------------
# Footstep burst factory
# ---------------------------------------------------------------------------

def _make_footstep_burst(
    seed:        int,
    material_id: int,
    intensity:   float,
    config:      dict,
) -> List[AudioBurst]:
    """Generate footstep burst(s) for a given material and intensity."""
    audio   = config.get("audio", {})
    foot_g  = float(audio.get("foot_gain", 1.0))
    rng     = _SeededRng(seed)
    bursts: List[AudioBurst] = []

    # --- main footstep burst ------------------------------------------------
    if material_id == MAT_DUST:
        dur   = float(audio.get("foot_dust_tail_ms", 120.0)) / 1000.0
        adsr  = ADSR(attack=0.002, decay=0.02, sustain=0.0, release=dur)
        flt   = BiquadFilter(cutoff_norm=0.06, mode="bp")   # 200-1.5k proxy
        burst = AudioBurst(rng=rng.fork(1), duration=dur + 0.03,
                           envelope=adsr, filter=flt,
                           noise_mode="pink", gain=intensity * foot_g * 0.6)

    elif material_id == MAT_ROCK:
        clk_str = float(audio.get("foot_rock_click_strength", 1.4))
        dur   = 0.04
        adsr  = ADSR(attack=0.001, decay=0.01, sustain=0.0, release=dur)
        flt   = BiquadFilter(cutoff_norm=0.25, mode="bp")
        burst = AudioBurst(rng=rng.fork(2), duration=dur + 0.01,
                           envelope=adsr, filter=flt,
                           noise_mode="white", gain=intensity * foot_g * clk_str)

    elif material_id == MAT_FRACT:
        dur   = 0.06
        adsr  = ADSR(attack=0.001, decay=0.015, sustain=0.0, release=dur)
        flt   = BiquadFilter(cutoff_norm=0.15, mode="bp")
        burst = AudioBurst(rng=rng.fork(3), duration=dur + 0.02,
                           envelope=adsr, filter=flt,
                           noise_mode="white", gain=intensity * foot_g)

    elif material_id == MAT_ICE:
        squeal = float(audio.get("foot_ice_squeal_strength", 0.8))
        dur   = 0.12
        adsr  = ADSR(attack=0.003, decay=0.03, sustain=0.3, release=dur * 0.5)
        flt   = BiquadFilter(cutoff_norm=0.30, mode="bp")   # 1-3k proxy
        burst = AudioBurst(rng=rng.fork(4), duration=dur,
                           envelope=adsr, filter=flt,
                           noise_mode="pink", gain=intensity * foot_g * squeal)

    else:  # MAT_DEBRIS or unknown
        dur   = 0.18
        adsr  = ADSR(attack=0.002, decay=0.05, sustain=0.0, release=dur)
        flt   = BiquadFilter(cutoff_norm=0.08, mode="bp")
        burst = AudioBurst(rng=rng.fork(5), duration=dur,
                           envelope=adsr, filter=flt,
                           noise_mode="pink", gain=intensity * foot_g * 0.7)

    bursts.append(burst)

    # --- low-frequency thump (weight) ---------------------------------------
    if intensity > 0.4 and material_id not in (MAT_ICE,):
        thump_adsr = ADSR(attack=0.001, decay=0.04, sustain=0.0, release=0.08)
        thump_flt  = BiquadFilter(cutoff_norm=0.02, mode="lp")   # 20-80 Hz proxy
        thump = AudioBurst(rng=rng.fork(10), duration=0.12,
                           envelope=thump_adsr, filter=thump_flt,
                           noise_mode="brown", gain=intensity * foot_g * 0.5)
        bursts.append(thump)

    # --- debris sifting cascade (LooseDebris only) --------------------------
    if material_id == MAT_DEBRIS:
        n_grains = max(1, int(rng.next_float() * 5 + 3))
        for i in range(n_grains):
            delay_frac = 0.1 + rng.next_float() * 0.5
            g_adsr = ADSR(attack=0.002, decay=0.02, sustain=0.0, release=0.06)
            g_flt  = BiquadFilter(cutoff_norm=0.07, mode="bp")
            g_burst = AudioBurst(rng=rng.fork(20 + i), duration=0.10,
                                 envelope=g_adsr, filter=g_flt,
                                 noise_mode="white",
                                 gain=intensity * foot_g * 0.25 * (1.0 - delay_frac))
            bursts.append(g_burst)

    # --- micro-clicks for fractured surface ---------------------------------
    if material_id == MAT_FRACT:
        n_micro = max(1, int(rng.next_float() * 4 + 2))
        for i in range(n_micro):
            mc_adsr = ADSR(attack=0.001, decay=0.005, sustain=0.0, release=0.008)
            mc_burst = AudioBurst(rng=rng.fork(30 + i), duration=0.015,
                                  envelope=mc_adsr, filter=None,
                                  noise_mode="white",
                                  gain=intensity * foot_g * 0.4)
            bursts.append(mc_burst)

    return bursts


# ---------------------------------------------------------------------------
# EventToAudioRouter
# ---------------------------------------------------------------------------

class EventToAudioRouter:
    """Translates AnimEvents and GeoEventSignals into AudioBursts."""

    def __init__(self, config: dict, global_seed: int) -> None:
        self._config       = config
        self._global_seed  = global_seed
        self._step_index   = 0

    def route_foot_plant(
        self,
        channel:    AudioChannel,
        foot_id:    int,
        intensity:  float,
        material_id: int,
        lat_cell:    int = 0,
        lon_cell:    int = 0,
    ) -> None:
        seed = (
            _hash32(self._global_seed)
            ^ _hash32(self._step_index)
            ^ _hash32(foot_id)
            ^ _hash32(lat_cell * 1000 + lon_cell)
            ^ _hash32(material_id)
        )
        bursts = _make_footstep_burst(seed, material_id, intensity, self._config)
        for b in bursts:
            channel.add_burst(b)
        self._step_index += 1
        _log.debug("FootPlant foot=%d mat=%d intensity=%.2f seed=%d bursts=%d",
                   foot_id, material_id, intensity, seed, len(bursts))

    def route_geo_signal(
        self,
        rumble_ch:  AudioChannel,
        impact_ch:  AudioChannel,
        signal:     GeoEventSignal,
        distance:   float,
    ) -> None:
        audio   = self._config.get("audio", {})
        if signal.phase == GeoEventPhase.IMPACT:
            self._spawn_geo_impact(impact_ch, signal, distance, audio)
        elif signal.phase == GeoEventPhase.PRE:
            self._spawn_geo_pre(rumble_ch, signal, distance, audio)

    # --- geo impact ---------------------------------------------------------

    def _spawn_geo_impact(
        self,
        ch:       AudioChannel,
        signal:   GeoEventSignal,
        distance: float,
        audio:    dict,
    ) -> None:
        gain   = float(audio.get("geo_impact_gain", 1.0))
        dist_k = math.exp(-max(0.0, distance - 10.0) / 200.0)
        seed   = _hash32(self._global_seed) ^ _hash32(id(signal))

        # Main low-frequency thump
        thump_adsr = ADSR(attack=0.001, decay=0.08, sustain=0.0, release=0.2)
        thump_flt  = BiquadFilter(cutoff_norm=0.015, mode="lp")
        thump = AudioBurst(
            rng=_SeededRng(seed),
            duration=0.35,
            envelope=thump_adsr,
            filter=thump_flt,
            noise_mode="brown",
            gain=signal.intensity * gain * dist_k * 2.0,
        )
        ch.add_burst(thump)

        # Debris cascade
        n = max(1, int(signal.intensity * 8 + 2))
        rng = _SeededRng(seed ^ 0xDEBBDE)
        for i in range(n):
            d_adsr = ADSR(attack=0.005, decay=0.05, sustain=0.1, release=0.15)
            d_flt  = BiquadFilter(cutoff_norm=0.07, mode="bp")
            d_b = AudioBurst(
                rng=rng.fork(i),
                duration=0.25,
                envelope=d_adsr,
                filter=d_flt,
                noise_mode="pink",
                gain=signal.intensity * gain * dist_k * 0.4,
            )
            ch.add_burst(d_b)

        _log.debug("GeoImpact seed=%d dist=%.0f intensity=%.2f bursts=%d",
                   seed, distance, signal.intensity, 1 + n)

    # --- geo pre-rumble -----------------------------------------------------

    def _spawn_geo_pre(
        self,
        ch:       AudioChannel,
        signal:   GeoEventSignal,
        distance: float,
        audio:    dict,
    ) -> None:
        gain   = float(audio.get("geo_pre_rumble_gain", 0.5))
        dist_k = math.exp(-max(0.0, distance - 10.0) / 500.0)
        seed   = _hash32(self._global_seed) ^ _hash32(id(signal)) ^ 0xAABBCC

        rumble_adsr = ADSR(attack=1.0, decay=0.5, sustain=0.8,
                           release=float(audio.get("geo_pre_rumble_release_s", 5.0)))
        rumble_flt  = BiquadFilter(cutoff_norm=0.005, mode="lp")
        rumble = AudioBurst(
            rng=_SeededRng(seed),
            duration=max(0.5, abs(signal.time_to_impact)),
            envelope=rumble_adsr,
            filter=rumble_flt,
            noise_mode="brown",
            gain=signal.intensity * gain * dist_k,
        )
        ch.add_burst(rumble)

        _log.debug("GeoPRE seed=%d dist=%.0f intensity=%.2f tti=%.1fs",
                   seed, distance, signal.intensity, signal.time_to_impact)


# ---------------------------------------------------------------------------
# Continuous synthesis helpers
# ---------------------------------------------------------------------------

class _WindSynth:
    """Continuous wind synthesis state."""

    def __init__(self, seed: int) -> None:
        self._pink  = NoiseGenerator(seed)
        self._brown = NoiseGenerator(seed ^ 0xFF01)
        self._lfo   = ToneGenerator(freq_hz=0.3, sample_rate=60.0)
        self._bp    = BiquadFilter(cutoff_norm=0.10, mode="bp")   # 200-2k proxy
        self._lp    = BiquadFilter(cutoff_norm=0.04, mode="lp")   # <120 Hz proxy
        self._whistle = BiquadFilter(cutoff_norm=0.35, mode="bp") # 1-4k proxy
        self._whistle_lfo = ToneGenerator(freq_hz=0.07, sample_rate=60.0)

    def tick(self, dt: float, wind_speed: float, gust_strength: float,
             storm_intensity: float, whistle_enable: bool,
             whistle_strength: float) -> float:
        """Return a wind synthesis sample."""
        norm_wind   = min(1.0, wind_speed / 40.0)
        norm_storm  = min(1.0, storm_intensity)

        # Main band-pass pink noise layer
        main = self._bp.process(self._pink.pink()) * norm_wind

        # Low-frequency storm layer
        low  = self._lp.process(self._brown.brown()) * norm_storm * 0.6

        # Gust modulation (LFO amplitude modulate main)
        lfo_val = (self._lfo.sine() + 1.0) * 0.5   # [0, 1]
        gust_mod = 1.0 + lfo_val * min(1.0, gust_strength / 10.0)
        main *= gust_mod

        # Whistle (strong wind only)
        whistle = 0.0
        if whistle_enable and norm_wind > 0.4:
            wlfo = (self._whistle_lfo.sine() + 1.0) * 0.5
            self._whistle.set_cutoff(0.30 + wlfo * 0.08)
            whistle = self._whistle.process(self._pink.white()) * (norm_wind - 0.4) * whistle_strength

        return main + low + whistle


class _StormSynth:
    """Continuous storm / whiteout hiss synthesis."""

    def __init__(self, seed: int) -> None:
        self._white = NoiseGenerator(seed ^ 0xAA0011)
        self._hp    = BiquadFilter(cutoff_norm=0.40, mode="hp")   # >2-5k proxy
        self._sine  = ToneGenerator(freq_hz=40.0, sample_rate=60.0)

    def tick(self, dt: float, dust: float, visibility: float,
             storm_intensity: float) -> float:
        hiss_gain = max(0.0, 1.0 - visibility) * storm_intensity
        hiss      = self._hp.process(self._white.white()) * hiss_gain

        # Low "pressure" sine
        freq  = 30.0 + storm_intensity * 30.0
        self._sine.set_freq(freq)
        pressure = self._sine.sine() * storm_intensity * 0.2

        return hiss + pressure


class _SuitSynth:
    """Breath + servo clicks + fabric noise for the suit channel."""

    def __init__(self, seed: int) -> None:
        self._breath_noise  = NoiseGenerator(seed ^ 0x5A17)
        self._breath_lp     = BiquadFilter(cutoff_norm=0.06, mode="lp")
        self._breath_phase  = 0.0
        self._fabric_noise  = NoiseGenerator(seed ^ 0x7A85)
        self._fabric_bp     = BiquadFilter(cutoff_norm=0.05, mode="bp")

    def tick(self, dt: float, effort: float, storm_intensity: float,
             wind_speed: float) -> float:
        # Breath: cycle depends on effort
        breath_rate = 0.2 + effort * 0.3   # Hz — faster with effort
        self._breath_phase = (self._breath_phase + breath_rate * dt) % 1.0
        breath_env  = 0.5 * (1.0 - math.cos(2.0 * math.pi * self._breath_phase))
        breath_gain = 0.08 + effort * 0.06
        breath      = self._breath_lp.process(self._breath_noise.white()) * breath_env * breath_gain

        # Fabric/friction (movement-driven, attenuated in storm)
        fabric_gain = max(0.0, 0.03 - storm_intensity * 0.03)
        fabric      = self._fabric_bp.process(self._fabric_noise.white()) * fabric_gain

        return breath + fabric


class _GeoRumbleSynth:
    """Continuous infrasound geo rumble (driven by active PRE signals)."""

    def __init__(self, seed: int) -> None:
        self._brown = NoiseGenerator(seed ^ 0x6E011)
        self._lp    = BiquadFilter(cutoff_norm=0.003, mode="lp")
        self._tone  = ToneGenerator(freq_hz=28.0, sample_rate=60.0)

    def tick(self, dt: float, rumble_level: float) -> float:
        if rumble_level <= 0.0:
            return 0.0
        noise = self._lp.process(self._brown.brown()) * 0.5
        tone  = self._tone.sine() * 0.5
        return (noise + tone) * rumble_level


# ---------------------------------------------------------------------------
# ProceduralAudioSystem — main public API
# ---------------------------------------------------------------------------

class ProceduralAudioSystem:
    """Stage 19 procedural audio without external assets.

    Parameters
    ----------
    config      : dict  (Config.all or subset containing "audio" key)
    global_seed : int   (world seed, for deterministic variations)
    """

    def __init__(self, config: Optional[dict] = None, global_seed: int = 42) -> None:
        self._config      = config or {}
        self._seed        = global_seed
        self._mixer       = AudioMixer(self._config)
        self._router      = EventToAudioRouter(self._config, global_seed)
        self._t           = 0.0

        # Continuous synthesis objects
        self._wind_synth   = _WindSynth(global_seed ^ 0xA1DF)
        self._storm_synth  = _StormSynth(global_seed ^ 0x5701)
        self._suit_synth   = _SuitSynth(global_seed ^ 0x5A17)
        self._rumble_synth = _GeoRumbleSynth(global_seed ^ 0x6E011)

        # Current PRE rumble level (decays between signals)
        self._rumble_level: float = 0.0

        audio = self._config.get("audio", {})
        self._whistle_enable:   bool  = bool(audio.get("wind_whistle_enable", True))
        self._whistle_strength: float = float(audio.get("wind_whistle_strength", 0.3))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        dt:               float,
        wind_speed:       float = 0.0,
        gust_strength:    float = 0.0,
        storm_intensity:  float = 0.0,
        dust:             float = 0.0,
        visibility:       float = 1.0,
        effort:           float = 0.0,
        cadence:          float = 1.0,
        character_state:  Optional[CharacterState] = None,
        anim_events:      Optional[List[AnimEvent]] = None,
        geo_signals:      Optional[List[GeoEventSignal]] = None,
        player_pos=None,
    ) -> None:
        """Advance the audio system one game-tick.

        Parameters
        ----------
        dt               : float  — elapsed game-seconds this frame
        wind_speed       : float  — m/s from ClimateSystem
        gust_strength    : float  — gust amplitude from ClimateSystem
        storm_intensity  : float  — 0..1 from ClimateSystem
        dust             : float  — 0..1 suspension
        visibility       : float  — 0..1 (1 = clear)
        effort           : float  — 0..1 character effort
        cadence          : float  — gait cadence multiplier
        character_state  : CharacterState or None
        anim_events      : list of AnimEvent (from ReflexSystem)
        geo_signals      : list of GeoEventSignal (from GeoEventSystem)
        player_pos       : Vec3 or None
        """
        self._t += dt
        anim_events  = anim_events  or []
        geo_signals  = geo_signals  or []

        # --- route anim events (foot plants handled externally too) ---------
        foot_ch = self._mixer.channels[AudioChannelName.FOOTSTEPS.value]
        for ev in anim_events:
            if ev.type in (AnimEventType.ON_BRACE, AnimEventType.ON_STUMBLE_STEP,
                           AnimEventType.ON_SLIP_RECOVER):
                # Suit servo click on brace/stumble
                suit_ch = self._mixer.channels[AudioChannelName.SUIT.value]
                adsr = ADSR(attack=0.001, decay=0.01, sustain=0.0, release=0.02)
                rng  = _SeededRng(_hash32(self._seed) ^ _hash32(int(ev.time * 1000)))
                burst = AudioBurst(rng=rng, duration=0.03, envelope=adsr,
                                   noise_mode="white", gain=0.15)
                suit_ch.add_burst(burst)

        # --- route geo signals ----------------------------------------------
        rumble_ch = self._mixer.channels[AudioChannelName.GEO_RUMBLE.value]
        impact_ch = self._mixer.channels[AudioChannelName.GEO_IMPACT.value]
        max_pre_level = 0.0
        for sig in geo_signals:
            dist = 0.0
            if player_pos is not None:
                dx = sig.position.x - player_pos.x
                dy = sig.position.y - player_pos.y
                dz = sig.position.z - player_pos.z
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            self._router.route_geo_signal(rumble_ch, impact_ch, sig, dist)
            if sig.phase == GeoEventPhase.PRE:
                attn = math.exp(-dist / 500.0)
                max_pre_level = max(max_pre_level, sig.intensity * attn)

        # Smooth rumble level
        self._rumble_level += (max_pre_level - self._rumble_level) * min(1.0, dt * 2.0)

        # --- continuous synthesis samples -----------------------------------
        cont: Dict[str, float] = {}
        cont[AudioChannelName.WIND.value] = self._wind_synth.tick(
            dt, wind_speed, gust_strength, storm_intensity,
            self._whistle_enable, self._whistle_strength,
        )
        cont[AudioChannelName.STORM.value] = self._storm_synth.tick(
            dt, dust, visibility, storm_intensity,
        )
        cont[AudioChannelName.SUIT.value] = self._suit_synth.tick(
            dt, effort, storm_intensity, wind_speed,
        )
        cont[AudioChannelName.GEO_RUMBLE.value] = self._rumble_synth.tick(
            dt, self._rumble_level,
        )

        # Update continuous gains on channels
        wind_ch  = self._mixer.channels[AudioChannelName.WIND.value]
        storm_ch = self._mixer.channels[AudioChannelName.STORM.value]
        suit_ch  = self._mixer.channels[AudioChannelName.SUIT.value]
        geo_r_ch = self._mixer.channels[AudioChannelName.GEO_RUMBLE.value]

        wind_ch._cont_gain  = 1.0
        storm_ch._cont_gain = 1.0
        suit_ch._cont_gain  = 1.0
        geo_r_ch._cont_gain = 1.0

        # --- advance mixer --------------------------------------------------
        self._mixer.tick(dt, cont)

    def trigger_foot_plant(
        self,
        foot:        int   = 0,
        intensity:   float = 1.0,
        material_id: int   = MAT_DUST,
        slide_vel:   float = 0.0,
        lat_cell:    int   = 0,
        lon_cell:    int   = 0,
    ) -> None:
        """Trigger a foot-plant burst directly (e.g. from ProceduralAnimationSystem).

        Parameters
        ----------
        foot        : int   0 = left, 1 = right
        intensity   : float step intensity 0..1
        material_id : int   MAT_DUST / MAT_ROCK / MAT_FRACT / MAT_ICE / MAT_DEBRIS
        slide_vel   : float current slide speed (adds sliding layer)
        lat_cell    : int   grid cell for seeding
        lon_cell    : int   grid cell for seeding
        """
        foot_ch = self._mixer.channels[AudioChannelName.FOOTSTEPS.value]
        self._router.route_foot_plant(foot_ch, foot, intensity, material_id,
                                      lat_cell, lon_cell)

        # Continuous sliding audio for Footsteps channel
        if slide_vel > 0.1:
            audio     = self._config.get("audio", {})
            slide_gain = min(1.0, slide_vel / 6.0) * float(audio.get("foot_gain", 1.0))
            adsr  = ADSR(attack=0.05, decay=0.1, sustain=0.8, release=0.3)
            seed  = _hash32(self._seed) ^ _hash32(int(slide_vel * 100))
            flt   = BiquadFilter(
                cutoff_norm=0.30 if material_id == MAT_ICE else 0.08,
                mode="bp"
            )
            burst = AudioBurst(rng=_SeededRng(seed), duration=0.5,
                               envelope=adsr, filter=flt,
                               noise_mode="pink", gain=slide_gain)
            foot_ch.add_burst(burst)

    def trigger_geo_signal(
        self,
        signal:    GeoEventSignal,
        player_pos=None,
    ) -> None:
        """Directly route a GeoEventSignal to the appropriate audio channel."""
        rumble_ch = self._mixer.channels[AudioChannelName.GEO_RUMBLE.value]
        impact_ch = self._mixer.channels[AudioChannelName.GEO_IMPACT.value]
        dist = 0.0
        if player_pos is not None:
            dx = signal.position.x - player_pos.x
            dy = signal.position.y - player_pos.y
            dz = signal.position.z - player_pos.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        self._router.route_geo_signal(rumble_ch, impact_ch, signal, dist)

    def get_rms_levels(self) -> Dict[str, float]:
        """Return per-channel RMS amplitude (0..1 range, useful for debug log)."""
        return self._mixer.get_rms()

    def get_debug_info(self) -> dict:
        rms = self.get_rms_levels()
        return {
            "time":          self._t,
            "rumble_level":  self._rumble_level,
            "rms_levels":    rms,
            "active_bursts": {
                k: len(ch._bursts)
                for k, ch in self._mixer.channels.items()
            },
        }
