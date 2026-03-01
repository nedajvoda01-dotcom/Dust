"""test_smoke.py — Stage 20 HeadlessSimTestRunner smoke tests.

These tests exercise the full GameBootstrap → runtime loop cycle without
requiring a display, GPU, or audio device.  They are safe to run in CI.

Tests
-----
1. TestBootToPlayable
   — Bootstrap produces a character with all core systems wired.
   — Character position is close to the planet surface.
   — No NaN values in the initial character state.
   — World identity seed matches the configured seed.

2. TestNoNanIn60s
   — Headless sim runs for 60 s without NaN/Inf in the character state.
   — SimulationScheduler advances climate and insolation ticks.
   — sim_time advances monotonically.
   — Job queue does not grow without bound.

3. TestSaveLoadRoundtrip
   — Run 30 s → shutdown (forces final save) → reload.
   — After reload: seed matches, lat/lon are within tolerance,
     sim_time is restored.

4. TestDeterminismShort
   — Two independent 30 s runs with the same seed produce identical
     climate tick counts, geo-event tick counts, and climate field hash.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.GameBootstrap import GameBootstrap
from src.math.PlanetMath import PlanetMath
from src.math.Vec3 import Vec3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bootstrap(
    seed: int = 42,
    save_dir: str | None = None,
    reset: bool = True,
) -> GameBootstrap:
    """Create and initialise a headless GameBootstrap for testing."""
    b    = GameBootstrap(headless=True)
    argv = ["--headless"]
    if seed != 42:
        argv += ["--seed", str(seed)]
    if reset:
        argv += ["--reset"]
    b.init(cli_args=argv, save_dir=save_dir)
    return b


def _run_sim(
    bootstrap: GameBootstrap,
    seconds: float,
    frame_dt: float = 0.016,
) -> None:
    """Drive *bootstrap* for *seconds* of real time."""
    elapsed = 0.0
    while elapsed < seconds:
        dt       = min(frame_dt, seconds - elapsed)
        bootstrap.tick(dt)
        elapsed += dt


def _climate_hash(climate) -> str:
    """Stable determinism hash over the climate temperature field."""
    if climate is None:
        return "none"
    parts: list[str] = []
    if hasattr(climate, "_temp"):
        # Sample every 16th cell to keep the hash compact but representative.
        parts.append(",".join(f"{v:.2f}" for v in climate._temp[::16]))
    if hasattr(climate, "_dust"):
        parts.append(",".join(f"{v:.4f}" for v in climate._dust[::16]))
    payload = "|".join(parts).encode()
    return hashlib.md5(payload).hexdigest()


# ---------------------------------------------------------------------------
# 1. TestBootToPlayable
# ---------------------------------------------------------------------------

class TestBootToPlayable(unittest.TestCase):
    """Bootstrap produces a valid, grounded character with all systems."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.b    = _make_bootstrap(seed=42, save_dir=self._tmp)

    def tearDown(self) -> None:
        self.b.shutdown()

    def test_character_created(self) -> None:
        self.assertIsNotNone(self.b.character)

    def test_core_systems_present(self) -> None:
        self.assertIsNotNone(self.b.astro,       "astro missing")
        self.assertIsNotNone(self.b.insolation,  "insolation missing")
        self.assertIsNotNone(self.b.climate,     "climate missing")
        self.assertIsNotNone(self.b.tectonic,    "tectonic missing")
        self.assertIsNotNone(self.b.geo_events,  "geo_events missing")
        self.assertIsNotNone(self.b.scheduler,   "scheduler missing")
        self.assertIsNotNone(self.b.clock,       "clock missing")
        self.assertIsNotNone(self.b.autosave,    "autosave missing")
        self.assertIsNotNone(self.b.identity,    "identity missing")

    def test_character_near_surface(self) -> None:
        """Character must be within 5 % of planet_radius from the surface."""
        pos = self.b.character.position
        r   = pos.length()
        self.assertGreater(r, self.b.planet_radius * 0.99,
                           "character is below the surface")
        self.assertLess(r, self.b.planet_radius * 1.05,
                        "character is too far above the surface")

    def test_identity_seed_matches_config(self) -> None:
        self.assertEqual(self.b.identity.seed, 42)

    def test_no_nan_in_initial_state(self) -> None:
        pos = self.b.character.position
        vel = self.b.character.velocity
        for val, name in [
            (pos.x, "pos.x"), (pos.y, "pos.y"), (pos.z, "pos.z"),
            (vel.x, "vel.x"), (vel.y, "vel.y"), (vel.z, "vel.z"),
        ]:
            self.assertTrue(math.isfinite(val), f"Non-finite {name}={val}")

    def test_is_running(self) -> None:
        self.assertTrue(self.b.is_running)


# ---------------------------------------------------------------------------
# 2. TestNoNanIn60s
# ---------------------------------------------------------------------------

class TestNoNanIn60s(unittest.TestCase):
    """60-second headless simulation stays numerically stable."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self.b    = _make_bootstrap(seed=99, save_dir=self._tmp)

    def tearDown(self) -> None:
        self.b.shutdown()

    def test_no_nan_60s(self) -> None:
        _run_sim(self.b, seconds=60.0, frame_dt=0.016)
        pos = self.b.character.position
        vel = self.b.character.velocity
        for val, name in [
            (pos.x, "pos.x"), (pos.y, "pos.y"), (pos.z, "pos.z"),
            (vel.x, "vel.x"), (vel.y, "vel.y"), (vel.z, "vel.z"),
        ]:
            self.assertTrue(math.isfinite(val), f"Non-finite {name}={val} after 60 s")

    def test_scheduler_climate_ticked(self) -> None:
        _run_sim(self.b, seconds=60.0, frame_dt=0.016)
        self.assertGreater(self.b.scheduler.climate_tick_count, 0,
                           "climate should have ticked at least once")

    def test_scheduler_insolation_ticked(self) -> None:
        _run_sim(self.b, seconds=60.0, frame_dt=0.016)
        self.assertGreater(self.b.scheduler.insolation_update_count, 0,
                           "insolation should have updated at least once")

    def test_sim_time_advanced(self) -> None:
        _run_sim(self.b, seconds=60.0, frame_dt=0.016)
        self.assertGreater(self.b.clock.sim_time, 0.0)

    def test_job_queue_bounded(self) -> None:
        """Pending job queue must not grow without bound."""
        _run_sim(self.b, seconds=30.0, frame_dt=0.016)
        pending = self.b.scheduler.job_queue.pending_count()
        self.assertLess(pending, 1000,
                        f"job queue has {pending} pending items (unbounded growth?)")


# ---------------------------------------------------------------------------
# 3. TestSaveLoadRoundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip(unittest.TestCase):
    """Save after 30 s, reload, and confirm key fields are restored."""

    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # --- First run ---
            b1 = _make_bootstrap(seed=7, save_dir=tmp, reset=True)
            _run_sim(b1, seconds=30.0, frame_dt=0.1)

            seed1     = b1.identity.seed
            pos1      = b1.character.position
            ll1       = PlanetMath.from_direction(pos1.normalized())
            sim_time1 = b1.clock.sim_time

            b1.shutdown()  # forces a final save

            # --- Second run: reload (no --reset) ---
            b2 = GameBootstrap(headless=True)
            b2.init(cli_args=["--headless"], save_dir=tmp)

            seed2      = b2.identity.seed
            pos2       = b2.character.position
            ll2        = PlanetMath.from_direction(pos2.normalized())
            sim_time2  = b2.clock.sim_time

            b2.shutdown()

            self.assertEqual(seed1, seed2, "seeds must match after reload")
            self.assertAlmostEqual(
                ll1.lat_rad, ll2.lat_rad, places=2,
                msg=f"lat mismatch: {math.degrees(ll1.lat_rad):.3f}° vs "
                    f"{math.degrees(ll2.lat_rad):.3f}°",
            )
            self.assertAlmostEqual(
                ll1.lon_rad, ll2.lon_rad, places=2,
                msg=f"lon mismatch: {math.degrees(ll1.lon_rad):.3f}° vs "
                    f"{math.degrees(ll2.lon_rad):.3f}°",
            )
            self.assertAlmostEqual(
                sim_time1, sim_time2, places=1,
                msg=f"simTime mismatch: {sim_time1:.2f}s vs {sim_time2:.2f}s",
            )


# ---------------------------------------------------------------------------
# 4. TestDeterminismShort
# ---------------------------------------------------------------------------

class TestDeterminismShort(unittest.TestCase):
    """Two runs with the same seed produce identical simulation outcomes."""

    def _run_and_capture(self, seed: int, seconds: float = 30.0) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            b = _make_bootstrap(seed=seed, save_dir=tmp, reset=True)
            _run_sim(b, seconds=seconds, frame_dt=0.1)

            climate_ticks = b.scheduler.climate_tick_count
            geo_ticks     = b.scheduler.geoevent_tick_count
            chash         = _climate_hash(b.climate)

            b.shutdown()
            return {
                "climate_ticks": climate_ticks,
                "geo_ticks":     geo_ticks,
                "climate_hash":  chash,
            }

    def test_determinism_same_seed(self) -> None:
        seed = 12345
        run1 = self._run_and_capture(seed)
        run2 = self._run_and_capture(seed)

        self.assertEqual(
            run1["climate_ticks"], run2["climate_ticks"],
            "Climate tick counts must be identical for the same seed",
        )
        self.assertEqual(
            run1["geo_ticks"], run2["geo_ticks"],
            "Geo-event tick counts must be identical for the same seed",
        )
        self.assertEqual(
            run1["climate_hash"], run2["climate_hash"],
            "Climate field hash must be identical for the same seed",
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
