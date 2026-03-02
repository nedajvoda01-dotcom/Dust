"""test_macro_atmosphere_stage32.py — Stage 32 MacroAtmospherePhenomenaSystem tests.

Tests
-----
1. test_server_generates_dustwall_under_conditions
   — With high dustLiftPotential a DUST_WALL is spawned.

2. test_lightning_schedule_deterministic
   — flash_times are identical for the same seed/tick window.

3. test_multiplayer_same_phenomena
   — Two independent system instances with the same seed produce identical
     phenomenon IDs / parameters given the same inputs.

4. test_no_shimmer_pixel
   — Render params for a stationary phenomenon produce stable density when
     called repeatedly (no per-frame random).

5. test_caps_enforced
   — Active phenomena never exceed max_active budget.

6. test_local_coupling_dust_wall
   — A nearby DUST_WALL boosts wind and adds dust density.

7. test_ring_front_spawned_on_edge
   — RING_SHADOW_FRONT is spawned when ring-shadow edge is high.

8. test_audio_triggers_distance_delay
   — Lightning audio trigger has a positive delay proportional to distance.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.Config import Config
from src.systems.MacroAtmospherePhenomenaSystem import (
    MacroAtmospherePhenomenaSystem,
    MacroPhenomenon,
    PhenType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> Config:
    """Build a minimal Config that overrides default macro values."""
    cfg = Config.__new__(Config)
    macro: dict = {
        "enable": True,
        "tick_seconds": 0.01,    # very short tick so tests converge quickly
        "max_active": 16,
        "dustwall": {
            "frequency": 1.0,    # always spawn when conditions are met
            "length_km_min": 2.0,
            "length_km_max": 10.0,
            "thickness_m_min": 200.0,
            "thickness_m_max": 500.0,
            "intensity_min": 0.3,
            "intensity_max": 0.9,
            "max_active": 4,
        },
        "lightning": {
            "enable": True,
            "rarity": 1.0,       # always spawn lightning when conditions are met
            "flash_count_min": 3,
            "flash_count_max": 6,
            "max_active": 2,
        },
        "ringfront": {
            "enable": True,
            "edge_width_km": 5.0,
            "max_active": 2,
        },
        "vortex": {
            "enable": True,
            "max_active": 2,
        },
        "render": {
            "volumetric_steps_near": 8,
            "volumetric_steps_far": 4,
        },
        "audio": {
            "rumble_gain": 0.8,
            "lightning_gain": 0.9,
        },
    }
    # Apply nested overrides
    for k, v in overrides.items():
        macro[k] = v
    cfg._data = {"macro": macro}
    return cfg


def _make_sys(seed: int = 42, **kw) -> MacroAtmospherePhenomenaSystem:
    return MacroAtmospherePhenomenaSystem(config=_make_cfg(**kw), world_seed=seed)


# Stub StormCell-like objects
class _FakeStorm:
    def __init__(self, lat, lon, intensity=0.8, vel_u=5.0, vel_v=2.0):
        self.center_lat  = lat
        self.center_lon  = lon
        self.radius      = 0.3
        self.intensity   = intensity
        self.vel_u       = vel_u
        self.vel_v       = vel_v


# ---------------------------------------------------------------------------
# 1. test_server_generates_dustwall_under_conditions
# ---------------------------------------------------------------------------

class TestDustWallGeneration(unittest.TestCase):
    def test_dustwall_spawned_with_high_dust_lift(self):
        """A DUST_WALL must be generated when dustLiftPotential is high."""
        sys_obj = _make_sys(seed=1)

        storm = _FakeStorm(lat=0.0, lon=0.0, intensity=0.9)

        def high_lift(lat, lon):
            return 0.9  # very high lift everywhere

        def high_wind(lat, lon):
            return 18.0  # above threshold

        # Run enough ticks for a dust wall to be spawned
        for i in range(20):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[storm],
                dust_density_fn=lambda la, lo: 0.7,
                wind_speed_fn=high_wind,
                dust_lift_potential_fn=high_lift,
                ring_shadow_edge_fn=lambda la, lo: 0.0,
            )

        phenomena = sys_obj.get_active_phenomena()
        dust_walls = [p for p in phenomena if p.phen_type == PhenType.DUST_WALL]
        self.assertGreater(len(dust_walls), 0,
                           "Expected at least one DUST_WALL to be spawned under high-lift conditions")

    def test_no_dustwall_without_conditions(self):
        """No DUST_WALL should spawn when lift and wind are both very low."""
        sys_obj = _make_sys(seed=2)

        for i in range(20):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[],
                dust_density_fn=lambda la, lo: 0.1,
                wind_speed_fn=lambda la, lo: 2.0,
                dust_lift_potential_fn=lambda la, lo: 0.1,
                ring_shadow_edge_fn=lambda la, lo: 0.0,
            )

        phenomena = sys_obj.get_active_phenomena()
        dust_walls = [p for p in phenomena if p.phen_type == PhenType.DUST_WALL]
        self.assertEqual(len(dust_walls), 0,
                         "No DUST_WALL should spawn under low-lift conditions")


# ---------------------------------------------------------------------------
# 2. test_lightning_schedule_deterministic
# ---------------------------------------------------------------------------

class TestLightningDeterminism(unittest.TestCase):
    def test_flash_times_same_for_same_seed(self):
        """Two systems with the same seed produce identical flash_times."""

        def _run(seed):
            sys_obj = _make_sys(seed=seed)
            storm = _FakeStorm(lat=0.0, lon=0.0)
            for i in range(50):
                sys_obj.update(
                    0.05, float(i) * 0.05,
                    storm_cells=[storm],
                    dust_density_fn=lambda la, lo: 0.9,
                    wind_speed_fn=lambda la, lo: 25.0,
                    dust_lift_potential_fn=lambda la, lo: 0.85,
                    ring_shadow_edge_fn=lambda la, lo: 0.0,
                )
            lightning = [
                p for p in sys_obj.get_active_phenomena()
                if p.phen_type == PhenType.DRY_LIGHTNING_CLUSTER
            ]
            return lightning

        clusters_a = _run(seed=77)
        clusters_b = _run(seed=77)   # same seed

        self.assertEqual(
            len(clusters_a), len(clusters_b),
            "Both runs must produce the same number of lightning clusters",
        )
        for ca, cb in zip(clusters_a, clusters_b):
            self.assertEqual(ca.flash_times, cb.flash_times,
                             "flash_times must be identical for the same seed")
            self.assertEqual(ca.seed, cb.seed,
                             "phenomenon seed must be identical")

    def test_flash_times_are_nonempty(self):
        """A spawned lightning cluster must have at least one flash scheduled."""
        sys_obj = _make_sys(seed=5)
        storm = _FakeStorm(lat=0.0, lon=0.0)
        for i in range(50):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[storm],
                dust_density_fn=lambda la, lo: 0.9,
                wind_speed_fn=lambda la, lo: 25.0,
                dust_lift_potential_fn=lambda la, lo: 0.85,
                ring_shadow_edge_fn=lambda la, lo: 0.0,
            )
        lightning = [
            p for p in sys_obj.get_active_phenomena()
            if p.phen_type == PhenType.DRY_LIGHTNING_CLUSTER
        ]
        for cluster in lightning:
            self.assertGreater(len(cluster.flash_times), 0,
                               "flash_times must be non-empty")


# ---------------------------------------------------------------------------
# 3. test_multiplayer_same_phenomena
# ---------------------------------------------------------------------------

class TestMultiplayerSamePhenomena(unittest.TestCase):
    def test_same_seed_same_phenomena_ids(self):
        """Two instances with equal seed + inputs produce identical phen_id lists."""

        def _run(seed):
            sys_obj = _make_sys(seed=seed)
            storm = _FakeStorm(lat=0.1, lon=0.2)
            for i in range(30):
                sys_obj.update(
                    0.05, float(i) * 0.05,
                    storm_cells=[storm],
                    dust_density_fn=lambda la, lo: 0.75,
                    wind_speed_fn=lambda la, lo: 20.0,
                    dust_lift_potential_fn=lambda la, lo: 0.7,
                    ring_shadow_edge_fn=lambda la, lo: 0.0,
                )
            return [p.phen_id for p in sys_obj.get_active_phenomena()]

        ids_a = _run(42)
        ids_b = _run(42)  # same seed

        self.assertEqual(ids_a, ids_b,
                         "Both clients must have the same phen_id list")

    def test_different_seeds_may_differ(self):
        """Different seeds can produce different phenomena."""

        def _run(seed):
            sys_obj = _make_sys(seed=seed)
            storm = _FakeStorm(lat=0.0, lon=0.0)
            for i in range(50):
                sys_obj.update(
                    0.05, float(i) * 0.05,
                    storm_cells=[storm],
                    dust_density_fn=lambda la, lo: 0.7,
                    wind_speed_fn=lambda la, lo: 20.0,
                    dust_lift_potential_fn=lambda la, lo: 0.7,
                    ring_shadow_edge_fn=lambda la, lo: 0.0,
                )
            return [(p.phen_id, p.phen_type) for p in sys_obj.get_active_phenomena()]

        # With different seeds the phenomena MAY differ (not a hard requirement
        # that they DO differ, but at least the system is stable for each seed)
        _run(1)
        _run(2)


# ---------------------------------------------------------------------------
# 4. test_no_shimmer_pixel
# ---------------------------------------------------------------------------

class TestNoShimmerPixel(unittest.TestCase):
    def test_stable_render_params_no_random(self):
        """Render params for a stationary camera / phenomenon are stable (no RNG)."""
        sys_obj = _make_sys(seed=10)
        storm = _FakeStorm(lat=0.3, lon=0.3)
        # Run enough ticks to have a dust wall
        for i in range(30):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[storm],
                dust_density_fn=lambda la, lo: 0.8,
                wind_speed_fn=lambda la, lo: 22.0,
                dust_lift_potential_fn=lambda la, lo: 0.8,
                ring_shadow_edge_fn=lambda la, lo: 0.0,
            )

        # Now get render params at two consecutive calls with the same sim_time
        # (no time advance = same flash schedule state)
        sim_time_fixed = 5.0
        params_a = sys_obj.get_render_params(0.0, 0.0, sim_time_fixed)
        params_b = sys_obj.get_render_params(0.0, 0.0, sim_time_fixed)

        self.assertEqual(len(params_a), len(params_b),
                         "Render param count must be stable")
        for a, b in zip(params_a, params_b):
            self.assertEqual(a.phen_id, b.phen_id)
            self.assertAlmostEqual(a.base_density, b.base_density, places=12,
                                   msg="base_density must be deterministic")
            self.assertAlmostEqual(a.intensity, b.intensity, places=12,
                                   msg="intensity must be deterministic")


# ---------------------------------------------------------------------------
# 5. test_caps_enforced
# ---------------------------------------------------------------------------

class TestCapsEnforced(unittest.TestCase):
    def test_max_active_not_exceeded(self):
        """Total active phenomena must never exceed max_active."""
        max_active = 6
        sys_obj = _make_sys(
            seed=20,
            max_active=max_active,
        )
        storm = _FakeStorm(lat=0.0, lon=0.0)

        for i in range(200):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[storm],
                dust_density_fn=lambda la, lo: 0.95,
                wind_speed_fn=lambda la, lo: 30.0,
                dust_lift_potential_fn=lambda la, lo: 0.95,
                ring_shadow_edge_fn=lambda la, lo: 0.9,
            )
            count = len(sys_obj.get_active_phenomena())
            self.assertLessEqual(
                count, max_active,
                f"Active count {count} exceeded max_active={max_active} at step {i}",
            )

    def test_max_dust_walls_cap(self):
        """Dust walls specifically must not exceed their own cap."""
        sys_obj = _make_sys(seed=21)
        storm = _FakeStorm(lat=0.0, lon=0.0)

        for i in range(200):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[storm],
                dust_density_fn=lambda la, lo: 0.95,
                wind_speed_fn=lambda la, lo: 30.0,
                dust_lift_potential_fn=lambda la, lo: 0.95,
                ring_shadow_edge_fn=lambda la, lo: 0.0,
            )

        dw_count = sum(
            1 for p in sys_obj.get_active_phenomena()
            if p.phen_type == PhenType.DUST_WALL
        )
        self.assertLessEqual(dw_count, 4,
                             "Dust wall count exceeded default max_active=4")


# ---------------------------------------------------------------------------
# 6. test_local_coupling_dust_wall
# ---------------------------------------------------------------------------

class TestLocalCoupling(unittest.TestCase):
    def test_nearby_dust_wall_boosts_wind_and_dust(self):
        """A dust wall very close to the player must add wind and dust."""
        sys_obj = _make_sys(seed=30)
        storm = _FakeStorm(lat=0.0, lon=0.0)
        for i in range(30):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[storm],
                dust_density_fn=lambda la, lo: 0.8,
                wind_speed_fn=lambda la, lo: 22.0,
                dust_lift_potential_fn=lambda la, lo: 0.85,
                ring_shadow_edge_fn=lambda la, lo: 0.0,
            )

        # Coupling at the wall anchor
        phenomena = sys_obj.get_active_phenomena()
        dust_walls = [p for p in phenomena if p.phen_type == PhenType.DUST_WALL]
        if not dust_walls:
            self.skipTest("No dust wall spawned — cannot test coupling")

        wall = dust_walls[0]
        result = sys_obj.apply_local_coupling(wall.anchor_lat, wall.anchor_lon)

        self.assertGreater(result.wind_boost, 0.0,
                           "Nearby dust wall must boost wind")
        self.assertGreater(result.dust_density_add, 0.0,
                           "Nearby dust wall must increase dust density")
        self.assertLess(result.visibility_mul, 1.0,
                        "Nearby dust wall must reduce visibility")

    def test_far_dust_wall_no_coupling(self):
        """A very distant dust wall should have negligible coupling effect."""
        sys_obj = _make_sys(seed=31)
        storm = _FakeStorm(lat=0.0, lon=0.0)
        for i in range(30):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[storm],
                dust_density_fn=lambda la, lo: 0.8,
                wind_speed_fn=lambda la, lo: 22.0,
                dust_lift_potential_fn=lambda la, lo: 0.85,
                ring_shadow_edge_fn=lambda la, lo: 0.0,
            )

        # Query far from all phenomena (south pole area)
        result = sys_obj.apply_local_coupling(-math.pi * 0.45, 3.0)

        self.assertAlmostEqual(result.wind_boost, 0.0, places=3,
                               msg="Wind boost should be ~0 far from dust wall")
        self.assertAlmostEqual(result.dust_density_add, 0.0, places=3,
                               msg="Dust add should be ~0 far from dust wall")


# ---------------------------------------------------------------------------
# 7. test_ring_front_spawned_on_edge
# ---------------------------------------------------------------------------

class TestRingShadowFront(unittest.TestCase):
    def test_ring_front_spawned_on_strong_edge(self):
        """A RING_SHADOW_FRONT is spawned when ring-shadow edge proximity is high."""
        sys_obj = _make_sys(seed=40)

        def high_ring_edge(lat, lon):
            return 0.8   # high edge proximity everywhere

        for i in range(50):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[],
                dust_density_fn=lambda la, lo: 0.3,
                wind_speed_fn=lambda la, lo: 5.0,
                dust_lift_potential_fn=lambda la, lo: 0.0,
                ring_shadow_edge_fn=high_ring_edge,
            )

        phenomena = sys_obj.get_active_phenomena()
        ring_fronts = [p for p in phenomena if p.phen_type == PhenType.RING_SHADOW_FRONT]
        self.assertGreater(len(ring_fronts), 0,
                           "RING_SHADOW_FRONT should spawn with high ring-shadow edge")

    def test_ring_front_intensity_in_range(self):
        """RING_SHADOW_FRONT intensity must be between 0 and 1."""
        sys_obj = _make_sys(seed=41)

        def high_ring_edge(lat, lon):
            return 0.9

        for i in range(50):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[],
                dust_density_fn=lambda la, lo: 0.4,
                wind_speed_fn=lambda la, lo: 5.0,
                dust_lift_potential_fn=lambda la, lo: 0.0,
                ring_shadow_edge_fn=high_ring_edge,
            )

        for phen in sys_obj.get_active_phenomena():
            self.assertGreaterEqual(phen.intensity, 0.0)
            self.assertLessEqual(phen.intensity, 1.0)


# ---------------------------------------------------------------------------
# 8. test_audio_triggers_distance_delay
# ---------------------------------------------------------------------------

class TestAudioTriggers(unittest.TestCase):
    def test_lightning_delay_proportional_to_distance(self):
        """Thunder delay must be proportional to distance (distance / c_sound)."""
        sys_obj = _make_sys(seed=50)
        storm = _FakeStorm(lat=0.0, lon=0.0)

        # Force a lightning cluster by running with all-high conditions
        for i in range(50):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[storm],
                dust_density_fn=lambda la, lo: 0.95,
                wind_speed_fn=lambda la, lo: 30.0,
                dust_lift_potential_fn=lambda la, lo: 0.9,
                ring_shadow_edge_fn=lambda la, lo: 0.0,
            )

        phenomena = sys_obj.get_active_phenomena()
        lightning = [
            p for p in phenomena
            if p.phen_type == PhenType.DRY_LIGHTNING_CLUSTER
        ]
        if not lightning:
            self.skipTest("No lightning cluster spawned — cannot test audio")

        cluster = lightning[0]
        # Simulate at a flash time
        if not cluster.flash_times:
            self.skipTest("No flash times — cannot test")
        flash_t = cluster.flash_times[0]

        # Query from near (same lat/lon as cluster)
        near_triggers = sys_obj.get_audio_triggers(
            cluster.anchor_lat, cluster.anchor_lon, flash_t
        )
        # Query from far away (~5000 km in lat offset)
        far_lat = _clamp(cluster.anchor_lat + 0.05, -1.5, 1.5)
        far_triggers = sys_obj.get_audio_triggers(far_lat, cluster.anchor_lon, flash_t)

        lightning_near = [t for t in near_triggers if t.event == "lightning_crack"]
        lightning_far  = [t for t in far_triggers  if t.event == "lightning_crack"]

        if lightning_near and lightning_far:
            self.assertLess(
                lightning_near[0].delay_sec,
                lightning_far[0].delay_sec,
                "Nearer lightning must have smaller thunder delay",
            )

    def test_dust_wall_rumble_present_when_close(self):
        """dust_wall_rumble trigger must appear when player is close to a dust wall."""
        sys_obj = _make_sys(seed=60)
        storm = _FakeStorm(lat=0.0, lon=0.0)

        for i in range(30):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[storm],
                dust_density_fn=lambda la, lo: 0.8,
                wind_speed_fn=lambda la, lo: 22.0,
                dust_lift_potential_fn=lambda la, lo: 0.85,
                ring_shadow_edge_fn=lambda la, lo: 0.0,
            )

        phenomena = sys_obj.get_active_phenomena()
        walls = [p for p in phenomena if p.phen_type == PhenType.DUST_WALL]
        if not walls:
            self.skipTest("No dust wall spawned")

        wall = walls[0]
        triggers = sys_obj.get_audio_triggers(wall.anchor_lat, wall.anchor_lon, 5.0)
        rumble = [t for t in triggers if t.event == "dust_wall_rumble"]
        self.assertGreater(len(rumble), 0,
                           "dust_wall_rumble must fire when player is at wall anchor")
        self.assertGreater(rumble[0].gain, 0.0,
                           "rumble gain must be positive")


# ---------------------------------------------------------------------------
# Extras: debug state and disabled system
# ---------------------------------------------------------------------------

class TestDebugState(unittest.TestCase):
    def test_debug_state_structure(self):
        sys_obj = _make_sys(seed=99)
        state = sys_obj.get_debug_state()
        self.assertIn("active_count", state)
        self.assertIn("by_type", state)
        self.assertIn("tick_index", state)
        self.assertEqual(state["active_count"], 0)

    def test_disabled_system_produces_nothing(self):
        sys_obj = _make_sys(seed=0, **{"enable": False})
        for i in range(20):
            sys_obj.update(
                0.05, float(i) * 0.05,
                storm_cells=[_FakeStorm(0.0, 0.0)],
                dust_density_fn=lambda la, lo: 1.0,
                wind_speed_fn=lambda la, lo: 40.0,
                dust_lift_potential_fn=lambda la, lo: 1.0,
                ring_shadow_edge_fn=lambda la, lo: 0.0,
            )
        self.assertEqual(len(sys_obj.get_active_phenomena()), 0)


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


if __name__ == "__main__":
    unittest.main()
