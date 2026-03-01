"""GameBootstrap — Stage 20 headless-compatible game lifecycle orchestrator.

Initialises all simulation systems in the correct causal order (as
specified in Section 4.2 of the Stage 20 spec), spawns the character on
the planet surface, and drives the runtime simulation loop.

No menu, no HUD, no tutorial.

Dev-only modes are activated exclusively via config keys or CLI args:
  --seed <int>        override world seed
  --reset             delete saved world and start fresh
  --timescale <float> override time.scale
  --no_audio          disable audio (no-op; placeholder for future)
  --headless          disable rendering (for smoke tests / CI)

Public API
----------
GameBootstrap(headless=False)
  .init(config_path, cli_args, save_dir)
  .tick(real_dt, movement_intent)   — advance one frame
  .shutdown()                       — final save + cleanup
  .is_spawned_grounded → bool       — True when character is on surface
  .character → CharacterPhysicalController
  .clock     → WorldClock
  .scheduler → SimulationScheduler
  .identity  → WorldIdentity
"""
from __future__ import annotations

import argparse
import math
from typing import Optional

from src.core.AutosaveManager import AutosaveManager
from src.core.Config import Config
from src.core.Logger import Logger
from src.core.PersistentStorage import PersistentStorage
from src.core.WorldClock import WorldClock
from src.core.WorldIdentity import WorldIdentity
from src.math.PlanetMath import PlanetMath
from src.math.Vec3 import Vec3
from src.planet.GeoFieldSampler import GeoFieldSampler
from src.planet.PlanetHeightProvider import PlanetHeightProvider
from src.planet.TectonicPlatesSystem import TectonicPlatesSystem
from src.systems.AstroSystem import AstroSystem
from src.systems.CharacterEnvironmentIntegration import CharacterEnvironmentIntegration
from src.systems.CharacterPhysicalController import (
    CharacterPhysicalController,
    EnvironmentSampler,
    IGroundSampler,
)
from src.systems.CharacterSpawnSystem import CharacterSpawnSystem
from src.systems.ClimateSystem import ClimateSystem
from src.systems.GeoEventSystem import GeoEventSystem
from src.systems.InsolationField import InsolationField
from src.systems.ReflexSystem import ReflexSystem
from src.systems.SimulationScheduler import SimulationScheduler

_TAG = "Bootstrap"
_SCHEMA_VERSION = 1


class GameBootstrap:
    """Orchestrates system initialisation, character spawn, and the
    runtime simulation loop for Dust.

    Use ``headless=True`` for smoke tests so that no display or audio
    is required.
    """

    def __init__(self, headless: bool = False) -> None:
        self.headless: bool = headless

        # Core
        self.config:    Optional[Config]             = None
        self.clock:     Optional[WorldClock]          = None
        self.storage:   Optional[PersistentStorage]   = None
        self.identity:  Optional[WorldIdentity]       = None
        self.autosave:  Optional[AutosaveManager]     = None
        self.scheduler: Optional[SimulationScheduler] = None

        # Simulation systems
        self.astro:       Optional[AstroSystem]        = None
        self.insolation:  Optional[InsolationField]    = None
        self.climate:     Optional[ClimateSystem]      = None
        self.tectonic:    Optional[TectonicPlatesSystem] = None
        self.geo_events:  Optional[GeoEventSystem]    = None
        self.height_provider: Optional[PlanetHeightProvider] = None

        # Character
        self.character:       Optional[CharacterPhysicalController]    = None
        self.reflex:          Optional[ReflexSystem]                    = None
        self.env_integration: Optional[CharacterEnvironmentIntegration] = None

        # Runtime state
        self.seed:          int   = 42
        self.planet_radius: float = 1000.0
        self.is_running:    bool  = False

        # Anti-NaN snapshots
        self._last_safe_pos: Optional[Vec3] = None
        self._last_safe_vel: Optional[Vec3] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init(
        self,
        config_path: Optional[str] = None,
        cli_args:    Optional[list] = None,
        save_dir:    Optional[str]  = None,
    ) -> None:
        """Initialise all systems in causal order.

        Parameters
        ----------
        config_path:
            Path to the main config JSON file (None → use project default).
        cli_args:
            List of CLI argument strings; if None, ``sys.argv[1:]`` is used.
        save_dir:
            Override save directory (None → ``<project_root>/saves``).
        """
        # ----------------------------------------------------------------
        # 1. Config
        # ----------------------------------------------------------------
        self.config = Config(config_path)
        schema_ver  = int(self.config.get("save", "schema_version", default=_SCHEMA_VERSION))

        # ----------------------------------------------------------------
        # Parse CLI overrides (dev only)
        # ----------------------------------------------------------------
        args = self._parse_cli(cli_args)

        # ----------------------------------------------------------------
        # 2. PersistentStorage
        # ----------------------------------------------------------------
        self.storage = PersistentStorage(save_dir)
        self.storage.init()

        if args.reset:
            Logger.info(_TAG, "--reset: clearing save data")
            self.storage.reset()

        # ----------------------------------------------------------------
        # 3. WorldIdentity
        # ----------------------------------------------------------------
        default_seed = (
            int(args.seed)
            if args.seed is not None
            else int(self.config.get("seed", "default", default=42))
        )
        self.identity = WorldIdentity.load_or_create(
            self.storage,
            default_seed  = default_seed,
            schema_version = schema_ver,
        )
        self.seed = self.identity.seed
        Logger.info(_TAG, f"seed={self.seed}  id={self.identity.world_id}")

        # ----------------------------------------------------------------
        # 4. WorldClock — restore simTime from previous save
        # ----------------------------------------------------------------
        time_scale = (
            float(args.timescale)
            if args.timescale is not None
            else float(self.config.get("time", "scale", default=1.0))
        )
        max_frame_dt = float(self.config.get("time", "max_frame_dt_clamp", default=0.1))
        self.clock = WorldClock(
            time_scale          = time_scale,
            max_frame_dt_clamp  = max_frame_dt,
        )

        # Temporarily load player state so we can restore simTime + position.
        _tmp_autosave    = AutosaveManager(self.storage, schema_version=schema_ver)
        saved_player     = _tmp_autosave.load_player_state(schema_ver)
        if saved_player is not None:
            restored_t = float(saved_player.get("simTime", 0.0))
            self.clock.sim_time  = restored_t
            self.clock.game_time = restored_t
            Logger.info(_TAG, f"Restored simTime={restored_t:.1f}s")

        # ----------------------------------------------------------------
        # 5. Planet params + height provider
        # ----------------------------------------------------------------
        self.planet_radius  = float(self.config.get("planet", "radius_units", default=1000.0))
        self.height_provider = PlanetHeightProvider(self.seed)

        # ----------------------------------------------------------------
        # 6. AstroSystem
        # ----------------------------------------------------------------
        self.astro = AstroSystem(self.config, seed=self.seed)
        self.astro.update(self.clock.sim_time)

        # ----------------------------------------------------------------
        # 7. InsolationField
        # ----------------------------------------------------------------
        self.insolation = InsolationField(self.config)
        self.insolation.force_full_update(
            self.clock.sim_time, self.astro, self.planet_radius
        )

        # ----------------------------------------------------------------
        # 8. ClimateSystem — warm up if no snapshot exists
        # ----------------------------------------------------------------
        self.climate = ClimateSystem(self.config, seed=self.seed)
        saved_climate = _tmp_autosave.load_climate_state(schema_ver)
        if saved_climate is None:
            warmup_s = int(self.config.get("boot", "warmup_seconds", default=10))
            Logger.info(_TAG, f"Climate warmup: {warmup_s}s …")
            for step in range(warmup_s):
                self.climate.update(1.0, insolation=self.insolation)
                if warmup_s >= 30 and (step + 1) % 10 == 0:
                    Logger.info(_TAG, f"  warmup {step + 1}/{warmup_s}s")

        # ----------------------------------------------------------------
        # 9. TectonicPlatesSystem (geology)
        # ----------------------------------------------------------------
        plate_count   = int(self.config.get("tectonic", "plate_count",   default=18))
        stress_rate   = float(self.config.get("tectonic", "stress_rate",   default=0.01))
        fracture_rate = float(self.config.get("tectonic", "fracture_rate", default=0.02))
        field_w       = int(self.config.get("tectonic", "field_width",    default=128))
        field_h       = int(self.config.get("tectonic", "field_height",   default=64))
        self.tectonic = TectonicPlatesSystem(
            seed          = self.seed,
            plate_count   = plate_count,
            stress_rate   = stress_rate,
            fracture_rate = fracture_rate,
            field_width   = field_w,
            field_height  = field_h,
        )
        self.tectonic.build()

        # ----------------------------------------------------------------
        # 10–11. GeoEventSystem + GeoEventLog
        # ----------------------------------------------------------------
        geo_sampler   = GeoFieldSampler(self.tectonic)
        pre_seconds   = float(self.config.get("geo", "event", "pre_seconds",  default=5.0))
        post_seconds  = float(self.config.get("geo", "event", "post_seconds", default=30.0))
        rate_minor    = float(self.config.get("geo", "events", "rate_minor_per_hour", default=20.0))
        rate_major    = float(self.config.get("geo", "events", "rate_major_per_hour", default=2.0))
        cooldown_min  = float(self.config.get("geo", "events", "cooldown_minutes_per_tile", default=5.0))
        self.geo_events = GeoEventSystem(
            geo_sampler               = geo_sampler,
            climate                   = self.climate,
            planet_radius             = self.planet_radius,
            height_provider           = self.height_provider,
            seed                      = self.seed,
            pre_seconds               = pre_seconds,
            post_seconds              = post_seconds,
            rate_minor_per_hour       = rate_minor,
            rate_major_per_hour       = rate_major,
            cooldown_minutes_per_tile = cooldown_min,
        )

        # ----------------------------------------------------------------
        # 12. SimulationScheduler
        # ----------------------------------------------------------------
        self.scheduler = SimulationScheduler(self.config, planet_radius=self.planet_radius)
        self.scheduler.astro      = self.astro
        self.scheduler.insolation = self.insolation
        self.scheduler.climate    = self.climate
        self.scheduler.geology    = self.tectonic
        self.scheduler.geo_events = self.geo_events

        # ----------------------------------------------------------------
        # 13. CharacterSpawnSystem
        # ----------------------------------------------------------------
        spawn_attempts = int(self.config.get("boot", "spawn_attempts",          default=64))
        slope_max_deg  = float(self.config.get("boot", "spawn_slope_max_deg",    default=25.0))
        stability_min  = float(self.config.get("boot", "spawn_stability_min",    default=0.3))
        avoid_storm    = float(self.config.get("boot", "spawn_avoid_storm_threshold", default=0.8))
        spawn_sys      = CharacterSpawnSystem(
            seed                  = self.seed,
            planet_radius         = self.planet_radius,
            spawn_attempts        = spawn_attempts,
            slope_max_deg         = slope_max_deg,
            stability_min         = stability_min,
            avoid_storm_threshold = avoid_storm,
            height_provider       = self.height_provider,
            tectonic_system       = self.tectonic,
            climate_system        = self.climate,
        )
        saved_lat = saved_player.get("lat_rad") if saved_player else None
        saved_lon = saved_player.get("lon_rad") if saved_player else None
        spawn_candidate = spawn_sys.spawn(saved_lat, saved_lon)
        spawn_pos       = spawn_sys.get_spawn_world_pos(spawn_candidate)

        # ----------------------------------------------------------------
        # 14. CharacterPhysicalController
        # ----------------------------------------------------------------
        env_sampler = EnvironmentSampler(
            climate    = self.climate,
            geo_events = self.geo_events,
        )
        self.reflex    = ReflexSystem(config=self.config)
        self.character = CharacterPhysicalController(
            position        = spawn_pos,
            planet_radius   = self.planet_radius,
            config          = self.config,
            env_sampler     = env_sampler,
            reflex_system   = self.reflex,
        )
        # Restore saved velocity if available
        if saved_player and "velocity" in saved_player:
            v = saved_player["velocity"]
            if len(v) == 3 and all(math.isfinite(c) for c in v):
                self.character.velocity = Vec3(v[0], v[1], v[2])

        # ----------------------------------------------------------------
        # 15. CharacterEnvironmentIntegration
        # ----------------------------------------------------------------
        self.env_integration = CharacterEnvironmentIntegration(
            config           = self.config,
            global_seed      = self.seed,
            character_id     = 0,
            climate          = self.climate,
            geo_event_system = self.geo_events,
            planet_radius    = self.planet_radius,
        )

        # ----------------------------------------------------------------
        # 16. AutosaveManager (real instance)
        # ----------------------------------------------------------------
        autosave_min      = float(self.config.get("save", "autosave_minutes",  default=2.0))
        save_on_geo       = bool(self.config.get("save", "on_geo_impact",      default=True))
        self.autosave     = AutosaveManager(
            storage            = self.storage,
            autosave_minutes   = autosave_min,
            save_on_geo_impact = save_on_geo,
            schema_version     = schema_ver,
        )

        # Snapshot safe initial state for NaN recovery
        self._last_safe_pos = Vec3(spawn_pos.x, spawn_pos.y, spawn_pos.z)
        self._last_safe_vel = Vec3.zero()

        self.is_running = True
        Logger.info(_TAG, "Bootstrap complete — character spawned, simulation active")

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def tick(
        self,
        real_dt:         float,
        movement_intent: Optional[Vec3] = None,
    ) -> None:
        """Advance the simulation by one frame.

        Parameters
        ----------
        real_dt:
            Wall-clock seconds elapsed since last frame.
        movement_intent:
            World-space movement direction (Vec3) or None for no input.
        """
        if not self.is_running:
            return
        assert self.clock and self.scheduler and self.autosave

        # 1. Clock
        self.clock.tick(real_dt)
        game_dt  = self.clock.game_dt
        sim_time = self.clock.sim_time

        if game_dt <= 0.0:
            return

        # 2–6. SimulationScheduler (astro → insol → climate → geo → events → jobs)
        player_pos = self.character.position if self.character else None
        self.scheduler.tick(game_dt, sim_time, player_pos)

        # 7. CharacterEnvironmentIntegration
        if self.env_integration is not None and self.character is not None:
            self.env_integration.update(
                self.character, self.reflex, game_dt, sim_time
            )

        # 8–9. Character physics
        if self.character is not None:
            desired_dir   = movement_intent if movement_intent is not None else Vec3.zero()
            walk_speed    = float(
                self.config.get("character", "walk_speed_units_per_s", default=1.8)
            )
            desired_speed = walk_speed if not desired_dir.is_near_zero() else 0.0
            self.character.update(
                game_dt,
                desired_dir   = desired_dir,
                desired_speed = desired_speed,
            )
            self._guard_nan()

        # 10. Autosave
        self.autosave.maybe_save(
            sim_time,
            character_controller = self.character,
            geo_event_system     = self.geo_events,
            climate_system       = self.climate,
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Force a final save and mark the bootstrap as stopped."""
        if self.autosave and self.clock:
            self.autosave.force_save(
                self.clock.sim_time,
                character_controller = self.character,
                geo_event_system     = self.geo_events,
                climate_system       = self.climate,
            )
        self.is_running = False
        Logger.info(_TAG, "Shutdown complete")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_spawned_grounded(self) -> bool:
        """True when the character is in contact with the planet surface."""
        from src.systems.CharacterPhysicalController import CharacterState
        if self.character is None:
            return False
        return self.character.state in (
            CharacterState.GROUNDED,
            CharacterState.SLIDING,
            CharacterState.BRACED,
            CharacterState.CROUCHED,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _guard_nan(self) -> None:
        """Detect NaN/Inf in character state and restore last safe snapshot."""
        c = self.character
        if c is None:
            return
        if _has_nan_vec3(c.position) or _has_nan_vec3(c.velocity):
            Logger.warn(_TAG, "NaN detected in character state — restoring last safe snapshot")
            if self._last_safe_pos is not None:
                c.position = Vec3(
                    self._last_safe_pos.x,
                    self._last_safe_pos.y,
                    self._last_safe_pos.z,
                )
            c.velocity = Vec3.zero()
        else:
            self._last_safe_pos = Vec3(c.position.x, c.position.y, c.position.z)

    @staticmethod
    def _parse_cli(argv: Optional[list] = None) -> argparse.Namespace:
        """Parse the subset of CLI arguments relevant to GameBootstrap."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--seed",       type=int,   default=None)
        parser.add_argument("--reset",      action="store_true")
        parser.add_argument("--timescale",  type=float, default=None)
        parser.add_argument("--no_audio",   action="store_true")
        parser.add_argument("--headless",   action="store_true")
        parser.add_argument("--debug_draw", action="store_true")
        args, _ = parser.parse_known_args(argv)
        return args


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _has_nan_vec3(v: Vec3) -> bool:
    """Return True if any component of *v* is not finite."""
    try:
        return (
            not math.isfinite(v.x)
            or not math.isfinite(v.y)
            or not math.isfinite(v.z)
        )
    except (AttributeError, TypeError):
        return True
