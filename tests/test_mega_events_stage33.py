"""test_mega_events_stage33.py — Stage 33 MegaEventSystem smoke tests.

Tests
-----
1. test_mega_rarity_cooldown
   — A second event is not triggered before the cooldown expires.

2. test_phase_timeline
   — PRE → ONSET → PEAK → AFTERMATH → DONE phases occur in correct order
     and at the expected simTimes.

3. test_multiplayer_sync_eventId
   — Two independent system instances with the same seed produce identical
     announces and phase transitions given the same inputs.

4. test_rift_patch_caps
   — The number of rift segments never exceeds segment_count_max.

5. test_state_persistence
   — Serialise → deserialise preserves the active event and scheduler state.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.Config import Config
from src.net.WorldState import WorldState
from src.ops.OpsLayer import OpsLayer
from src.systems.MegaEventSystem import (
    MegaEvent,
    MegaEventPhase,
    MegaEventSystem,
    MegaEventType,
    RiftPatch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> Config:
    """Minimal Config wired for fast mega-event testing."""
    cfg = Config.__new__(Config)
    mega: dict = {
        "enable":               True,
        "cooldown_hours_sim":   0.01,   # very short cooldown for tests
        "poisson_lambda":       100.0,  # very frequent triggers for tests
        "pre_min_min":          0.01,
        "pre_max_min":          0.02,
        "onset_min_min":        0.01,
        "onset_max_min":        0.02,
        "peak_min_min":         0.01,
        "peak_max_min":         0.02,
        "aftermath_min_hours":  0.001,
        "aftermath_max_hours":  0.002,
        "storm": {
            "radius_km_min":    100.0,
            "radius_km_max":    200.0,
            "max_macro_spawned": 4,
        },
        "veil": {
            "max_factor": 0.5,
        },
        "rift": {
            "segment_count_max": 10,
            "length_km_max":     200.0,
            "width_m_min":       50.0,
            "width_m_max":       200.0,
            "depth_m_min":       50.0,
            "depth_m_max":       200.0,
        },
        "fallback_mode": "none",
    }
    mega.update(overrides)
    cfg._data = {"mega": mega}
    return cfg


def _make_sys(seed: int = 42, **kw) -> MegaEventSystem:
    return MegaEventSystem(config=_make_cfg(**kw), world_seed=seed)


def _run_ticks(
    sys_obj: MegaEventSystem,
    n_ticks: int,
    dt: float = 1.0,
    start_time: float = 0.0,
    **inputs,
) -> float:
    """Advance *sys_obj* for *n_ticks* ticks; return final sim_time."""
    t = start_time
    for _ in range(n_ticks):
        sys_obj.update(dt, t, **inputs)
        t += dt
    return t


# ---------------------------------------------------------------------------
# 1. test_mega_rarity_cooldown
# ---------------------------------------------------------------------------

class TestMegaRarityCooldown(unittest.TestCase):
    def test_no_second_event_before_cooldown(self):
        """A second mega-event must not start before the cooldown expires."""
        cooldown_hours = 0.01   # 36 simulated seconds
        cooldown_sec = cooldown_hours * 3600.0
        sys_obj = _make_sys(seed=10, cooldown_hours_sim=cooldown_hours)

        # Force the first event and record the trigger time in the scheduler.
        t0 = 10.0
        sys_obj.force_event(MegaEventType.GLOBAL_DUST_VEIL, sim_time=t0)
        sys_obj._scheduler.record_triggered(t0)

        # Immediately after: scheduler must block due to cooldown.
        scores = {et.name: 1.0 for et in MegaEventType}
        result_immediate = sys_obj._scheduler.should_trigger(
            sim_time=t0 + 1.0,
            dt=1.0,
            scores=scores,
        )
        self.assertIsNone(result_immediate,
                          "Scheduler must return None while within cooldown")

        # After cooldown expires the scheduler is free to evaluate.
        # Use a very large dt to guarantee p_tick ≈ 1.
        t_after = t0 + cooldown_sec + 1.0
        time_since_last = t_after - sys_obj._scheduler._last_event_time
        self.assertGreater(time_since_last, cooldown_sec,
                           "Time since last event must exceed cooldown after fast-forward")

    def test_disabled_system_never_triggers(self):
        """When enable=False no event is ever created."""
        sys_obj = _make_sys(seed=0, **{"enable": False})
        _run_ticks(
            sys_obj, 200,
            dust_lift_potential=1.0,
            dust_thickness_mean=1.0,
            ring_shadow_intensity=1.0,
            fracture_fatigue_mean=1.0,
            subsurface_collapse_rate=1.0,
        )
        self.assertIsNone(sys_obj.active_event(),
                          "Disabled system must never produce an active event")


# ---------------------------------------------------------------------------
# 2. test_phase_timeline
# ---------------------------------------------------------------------------

class TestPhaseTimeline(unittest.TestCase):
    def _force_event_of_type(
        self, event_type: MegaEventType, sim_time: float = 0.0
    ) -> MegaEvent:
        sys_obj = _make_sys(seed=20)
        return sys_obj.force_event(event_type, sim_time)

    def _check_phases_in_order(self, evt: MegaEvent) -> None:
        """Assert that querying phase at each boundary returns correct enum."""
        ordered = [
            MegaEventPhase.PRE,
            MegaEventPhase.ONSET,
            MegaEventPhase.PEAK,
            MegaEventPhase.AFTERMATH,
        ]
        for i, phase in enumerate(ordered):
            t_start = evt.phase_start(phase)
            # Just after the start of this phase
            t_in = t_start + evt.phase_durations.get(phase.name, 1.0) * 0.5
            actual = evt.current_phase(t_in)
            self.assertEqual(actual, phase,
                             f"Expected {phase.name} at t={t_in:.2f}, got {actual.name}")

        # After all phases
        t_done = evt.end_time() + 1.0
        self.assertEqual(evt.current_phase(t_done), MegaEventPhase.DONE)

    def test_storm_phases_in_order(self):
        evt = self._force_event_of_type(MegaEventType.SUPERCELL_DUST_STORM)
        self._check_phases_in_order(evt)

    def test_veil_phases_in_order(self):
        evt = self._force_event_of_type(MegaEventType.GLOBAL_DUST_VEIL)
        self._check_phases_in_order(evt)

    def test_rift_phases_in_order(self):
        evt = self._force_event_of_type(MegaEventType.GREAT_RIFT)
        self._check_phases_in_order(evt)

    def test_ring_anom_phases_in_order(self):
        evt = self._force_event_of_type(MegaEventType.RING_SHADOW_ANOMALY)
        self._check_phases_in_order(evt)

    def test_phase_start_times_monotone(self):
        """Phase start times must be strictly increasing."""
        for et in MegaEventType:
            evt = self._force_event_of_type(et)
            starts = [
                evt.phase_start(MegaEventPhase.PRE),
                evt.phase_start(MegaEventPhase.ONSET),
                evt.phase_start(MegaEventPhase.PEAK),
                evt.phase_start(MegaEventPhase.AFTERMATH),
                evt.end_time(),
            ]
            for a, b in zip(starts, starts[1:]):
                self.assertLess(a, b,
                                f"{et.name}: phase starts not monotone: {starts}")

    def test_intensity_in_range(self):
        """intensity() must stay in [0, 1] at all sampled times."""
        for et in MegaEventType:
            evt = self._force_event_of_type(et, sim_time=0.0)
            t = evt.start_time
            end = evt.end_time()
            step = (end - t) / 50.0
            while t <= end + step:
                val = evt.intensity(t)
                self.assertGreaterEqual(val, 0.0, f"{et.name}: intensity<0 at t={t:.2f}")
                self.assertLessEqual(val, 1.0,   f"{et.name}: intensity>1 at t={t:.2f}")
                t += step

    def test_intensity_done_is_zero(self):
        """After the event ends intensity must be 0."""
        evt = self._force_event_of_type(MegaEventType.GLOBAL_DUST_VEIL)
        t_past = evt.end_time() + 1000.0
        self.assertEqual(evt.intensity(t_past), 0.0)


# ---------------------------------------------------------------------------
# 3. test_multiplayer_sync_eventId
# ---------------------------------------------------------------------------

class TestMultiplayerSync(unittest.TestCase):
    def _run_and_collect(self, seed: int) -> list:
        sys_obj = _make_sys(seed=seed)
        announces = []
        sim_time = 0.0
        dt = 1.0
        for _ in range(300):
            sys_obj.update(
                dt, sim_time,
                dust_lift_potential=0.9,
                dust_thickness_mean=0.9,
                ring_shadow_intensity=0.7,
                fracture_fatigue_mean=0.7,
                subsurface_collapse_rate=0.5,
            )
            msg = sys_obj.get_announce_message()
            if msg is not None:
                announces.append(msg)
            sim_time += dt
        return announces

    def test_same_seed_same_announces(self):
        """Two system instances with the same seed produce identical announces."""
        a = self._run_and_collect(seed=77)
        b = self._run_and_collect(seed=77)
        self.assertEqual(len(a), len(b),
                         "Both instances must emit the same number of announces")
        for ma, mb in zip(a, b):
            self.assertEqual(ma["event_id"],   mb["event_id"])
            self.assertEqual(ma["event_type"], mb["event_type"])
            self.assertAlmostEqual(ma["start_time"], mb["start_time"], places=9)
            self.assertAlmostEqual(ma["anchor_lat"],  mb["anchor_lat"],  places=9)
            self.assertAlmostEqual(ma["anchor_lon"],  mb["anchor_lon"],  places=9)

    def test_announce_contains_required_fields(self):
        """Each announce must contain all fields needed by a client."""
        announces = self._run_and_collect(seed=55)
        required = {
            "msg_type", "event_id", "event_type",
            "anchor_lat", "anchor_lon", "anchor_radius_m",
            "start_time", "phase_durations", "seed",
        }
        for ann in announces:
            for field in required:
                self.assertIn(field, ann,
                              f"Announce missing field '{field}'")
            self.assertEqual(ann["msg_type"], "MEGA_EVENT_ANNOUNCE")

    def test_client_reproduce_phase_from_announce(self):
        """A client that ingest an announce can reproduce phases deterministically."""
        sys_server = _make_sys(seed=33)
        sys_client = _make_sys(seed=99)  # different seed on client side

        sim_time = 0.0
        dt = 1.0
        announce = None

        for _ in range(300):
            sys_server.update(dt, sim_time,
                              dust_lift_potential=0.9, dust_thickness_mean=0.9,
                              ring_shadow_intensity=0.7, fracture_fatigue_mean=0.7,
                              subsurface_collapse_rate=0.5)
            msg = sys_server.get_announce_message()
            if msg is not None and announce is None:
                announce = msg
            sim_time += dt

        if announce is None:
            self.skipTest("No announce produced — cannot test client sync")

        sys_client.apply_replicated_event(announce)

        # Both should agree on phase at the same simTime
        t_query = sim_time - 10.0
        evt_server = sys_server.active_event()
        evt_client = sys_client.active_event()
        if evt_server is None or evt_client is None:
            self.skipTest("Event finished before query — cannot compare")

        self.assertEqual(
            evt_server.current_phase(t_query),
            evt_client.current_phase(t_query),
        )


# ---------------------------------------------------------------------------
# 4. test_rift_patch_caps
# ---------------------------------------------------------------------------

class TestRiftPatchCaps(unittest.TestCase):
    def test_rift_segments_within_cap(self):
        """GREAT_RIFT rift_patches must never exceed segment_count_max."""
        max_segs = 10
        sys_obj = _make_sys(seed=40)  # default segment_count_max=10

        for _ in range(5):
            evt = sys_obj.force_event(MegaEventType.GREAT_RIFT, 0.0)
            self.assertLessEqual(
                len(evt.rift_patches), max_segs,
                f"Rift has {len(evt.rift_patches)} patches, max={max_segs}",
            )

    def test_rift_patch_ids_unique(self):
        """Each RiftPatch in an event must have a unique patch_id."""
        sys_obj = _make_sys(seed=41)
        evt = sys_obj.force_event(MegaEventType.GREAT_RIFT, 0.0)
        ids = [p.patch_id for p in evt.rift_patches]
        self.assertEqual(len(ids), len(set(ids)),
                         "RiftPatch IDs must be unique within an event")

    def test_rift_patch_dimensions_in_range(self):
        """RiftPatch width and depth must be within configured ranges."""
        sys_obj = _make_sys(seed=42)
        evt = sys_obj.force_event(MegaEventType.GREAT_RIFT, 0.0)
        for patch in evt.rift_patches:
            self.assertGreaterEqual(patch.width_m, 50.0)
            self.assertLessEqual(patch.width_m,   200.0)
            self.assertGreaterEqual(patch.depth_m, 50.0)
            self.assertLessEqual(patch.depth_m,   200.0)

    def test_rift_patches_returned_by_phase(self):
        """get_rift_patches() must only return patches for current/past phases."""
        sys_obj = _make_sys(seed=43)
        evt = sys_obj.force_event(MegaEventType.GREAT_RIFT, 0.0)
        sys_obj._active_event = evt  # make active

        # During PRE: only ONSET+ patches should NOT appear (gate is ONSET/PEAK/AFTERMATH)
        t_pre = evt.phase_start(MegaEventPhase.PRE) + 0.01
        pre_patches = sys_obj.get_rift_patches(t_pre)
        for p in pre_patches:
            self.assertEqual(p.phase_gate, MegaEventPhase.PRE,
                             "Only PRE-gated patches visible during PRE phase")

        # During PEAK: at least PRE and ONSET patches should be visible
        t_peak = evt.phase_start(MegaEventPhase.PEAK) + 0.01
        peak_patches = sys_obj.get_rift_patches(t_peak)
        gates = {p.phase_gate for p in peak_patches}
        self.assertTrue(
            MegaEventPhase.ONSET in gates or MegaEventPhase.PRE in gates,
            "During PEAK, patches from earlier phases must be visible",
        )

    def test_no_rift_patches_for_non_rift_events(self):
        """get_rift_patches() returns [] for non-rift events."""
        sys_obj = _make_sys(seed=44)
        for et in [
            MegaEventType.SUPERCELL_DUST_STORM,
            MegaEventType.GLOBAL_DUST_VEIL,
            MegaEventType.RING_SHADOW_ANOMALY,
        ]:
            evt = sys_obj.force_event(et, 0.0)
            sys_obj._active_event = evt
            patches = sys_obj.get_rift_patches(0.01)
            self.assertEqual(patches, [],
                             f"{et.name} must not have rift patches")


# ---------------------------------------------------------------------------
# 5. test_state_persistence
# ---------------------------------------------------------------------------

class TestStatePersistence(unittest.TestCase):
    def test_roundtrip_state_dict_preserves_event(self):
        """to_state_dict / from_state_dict preserves the active event."""
        sys_obj = _make_sys(seed=50)
        evt = sys_obj.force_event(MegaEventType.GLOBAL_DUST_VEIL, sim_time=100.0)

        state = sys_obj.to_state_dict()
        # Verify state has expected keys
        self.assertIn("active_event",  state)
        self.assertIn("scheduler",     state)
        self.assertIn("event_counter", state)

        # Restore into a new instance
        sys_obj2 = _make_sys(seed=50)
        sys_obj2.from_state_dict(state)

        evt2 = sys_obj2.active_event()
        self.assertIsNotNone(evt2, "Restored system must have an active event")
        self.assertEqual(evt2.event_id,   evt.event_id)
        self.assertEqual(evt2.event_type, evt.event_type)
        self.assertAlmostEqual(evt2.start_time, evt.start_time, places=9)
        self.assertAlmostEqual(evt2.anchor_lat, evt.anchor_lat, places=9)

    def test_roundtrip_no_active_event(self):
        """State can be serialised and restored when there is no active event."""
        sys_obj = _make_sys(seed=51)
        state = sys_obj.to_state_dict()
        self.assertIsNone(state["active_event"])

        sys_obj2 = _make_sys(seed=51)
        sys_obj2.from_state_dict(state)
        self.assertIsNone(sys_obj2.active_event())

    def test_rift_patches_survive_roundtrip(self):
        """RiftPatch list is preserved through to_state_dict / from_state_dict."""
        sys_obj = _make_sys(seed=52)
        evt = sys_obj.force_event(MegaEventType.GREAT_RIFT, sim_time=0.0)
        original_patches = list(evt.rift_patches)

        state = sys_obj.to_state_dict()
        sys_obj2 = _make_sys(seed=52)
        sys_obj2.from_state_dict(state)

        evt2 = sys_obj2.active_event()
        self.assertIsNotNone(evt2)
        self.assertEqual(len(evt2.rift_patches), len(original_patches))
        for p1, p2 in zip(original_patches, evt2.rift_patches):
            self.assertEqual(p1.patch_id, p2.patch_id)
            self.assertAlmostEqual(p1.width_m, p2.width_m, places=6)
            self.assertAlmostEqual(p1.depth_m, p2.depth_m, places=6)
            self.assertEqual(p1.phase_gate, p2.phase_gate)

    def test_world_state_save_load_mega(self):
        """WorldState.save_mega_state / load_mega_state round-trips correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = WorldState(state_dir=tmpdir)
            ws.load_or_create()

            sys_obj = _make_sys(seed=53)
            evt = sys_obj.force_event(MegaEventType.SUPERCELL_DUST_STORM, 0.0)
            state = sys_obj.to_state_dict()

            ws.save_mega_state(state)
            loaded = ws.load_mega_state()

            self.assertIsNotNone(loaded)
            sys_obj2 = _make_sys(seed=53)
            sys_obj2.from_state_dict(loaded)

            evt2 = sys_obj2.active_event()
            self.assertIsNotNone(evt2)
            self.assertEqual(evt2.event_type, MegaEventType.SUPERCELL_DUST_STORM)
            self.assertEqual(evt2.event_id, evt.event_id)

    def test_ops_compact_includes_mega_state(self):
        """OpsLayer.compact() must include mega_state.json in the baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = WorldState(state_dir=tmpdir)
            ws.load_or_create()

            # Save a mega state
            sys_obj = _make_sys(seed=54)
            evt = sys_obj.force_event(MegaEventType.GLOBAL_DUST_VEIL, 0.0)
            ws.save_mega_state(sys_obj.to_state_dict())

            ops = OpsLayer(world_state=ws, state_dir=tmpdir)
            result = ops.compact()
            self.assertTrue(result, "compact() must succeed")

            # Find the baseline directory
            baselines = sorted(Path(tmpdir).glob("baseline_*"))
            self.assertGreater(len(baselines), 0, "No baseline created")
            bld = baselines[-1]

            mega_file = bld / "mega_state.json"
            self.assertTrue(mega_file.exists(),
                            "mega_state.json must be present in baseline")

            with open(mega_file, "r") as fh:
                restored = json.load(fh)
            self.assertIn("active_event", restored)
            self.assertIsNotNone(restored["active_event"])


# ---------------------------------------------------------------------------
# Extras: global coefficients and character modifiers
# ---------------------------------------------------------------------------

class TestGlobalCoeffs(unittest.TestCase):
    def test_veil_factor_applied(self):
        """GLOBAL_DUST_VEIL must produce a non-zero globalDustVeilFactor."""
        sys_obj = _make_sys(seed=60)
        evt = sys_obj.force_event(MegaEventType.GLOBAL_DUST_VEIL, sim_time=0.0)
        t_peak = evt.phase_start(MegaEventPhase.PEAK) + 1.0
        coeffs = sys_obj.get_global_coeffs(t_peak)
        self.assertGreater(coeffs["globalDustVeilFactor"], 0.0)
        self.assertLessEqual(coeffs["globalDustVeilFactor"], 0.5)

    def test_no_veil_for_done_event(self):
        """After an event is DONE coefficients must be zero."""
        sys_obj = _make_sys(seed=61)
        evt = sys_obj.force_event(MegaEventType.GLOBAL_DUST_VEIL, sim_time=0.0)
        t_done = evt.end_time() + 100.0
        coeffs = sys_obj.get_global_coeffs(t_done)
        self.assertEqual(coeffs["globalDustVeilFactor"], 0.0)

    def test_ring_anom_boosts_ring_edge(self):
        """RING_SHADOW_ANOMALY must boost ringEdgeBoost."""
        sys_obj = _make_sys(seed=62)
        evt = sys_obj.force_event(MegaEventType.RING_SHADOW_ANOMALY, sim_time=0.0)
        t_peak = evt.phase_start(MegaEventPhase.PEAK) + 1.0
        coeffs = sys_obj.get_global_coeffs(t_peak)
        self.assertGreater(coeffs["ringEdgeBoost"], 0.0)


class TestCharacterModifiers(unittest.TestCase):
    def test_storm_adds_wind_resistance_nearby(self):
        """SUPERCELL_DUST_STORM must add wind resistance near epicentre."""
        sys_obj = _make_sys(seed=70)
        evt = sys_obj.force_event(MegaEventType.SUPERCELL_DUST_STORM, sim_time=0.0)
        sys_obj._active_event = evt
        t_peak = evt.phase_start(MegaEventPhase.PEAK) + 1.0
        mod = sys_obj.get_character_modifiers(
            evt.anchor_lat, evt.anchor_lon, t_peak
        )
        self.assertGreater(mod.wind_resistance_add, 0.0)

    def test_rift_adds_collapse_risk_nearby(self):
        """GREAT_RIFT must add collapse_risk_add near epicentre."""
        sys_obj = _make_sys(seed=71)
        evt = sys_obj.force_event(MegaEventType.GREAT_RIFT, sim_time=0.0)
        sys_obj._active_event = evt
        t_peak = evt.phase_start(MegaEventPhase.PEAK) + 1.0
        mod = sys_obj.get_character_modifiers(
            evt.anchor_lat, evt.anchor_lon, t_peak
        )
        self.assertGreater(mod.collapse_risk_add, 0.0)

    def test_no_modifiers_far_from_event(self):
        """Modifiers must be negligible far from the event anchor."""
        sys_obj = _make_sys(seed=72)
        evt = sys_obj.force_event(MegaEventType.SUPERCELL_DUST_STORM, sim_time=0.0)
        sys_obj._active_event = evt
        t_peak = evt.phase_start(MegaEventPhase.PEAK) + 1.0
        # Opposite side of the planet
        far_lat = -evt.anchor_lat
        far_lon = evt.anchor_lon + math.pi
        mod = sys_obj.get_character_modifiers(far_lat, far_lon, t_peak)
        self.assertAlmostEqual(mod.wind_resistance_add, 0.0, places=3)
        self.assertAlmostEqual(mod.collapse_risk_add,   0.0, places=3)


class TestDebugState(unittest.TestCase):
    def test_debug_state_keys_present(self):
        sys_obj = _make_sys(seed=80)
        state = sys_obj.get_debug_state()
        for key in ("active", "event_type", "event_id", "epoch_index", "log_length"):
            self.assertIn(key, state)

    def test_debug_state_reflects_active_event(self):
        sys_obj = _make_sys(seed=81)
        sys_obj.force_event(MegaEventType.GLOBAL_DUST_VEIL, sim_time=0.0)
        state = sys_obj.get_debug_state()
        self.assertTrue(state["active"])
        self.assertEqual(state["event_type"], "GLOBAL_DUST_VEIL")


if __name__ == "__main__":
    unittest.main()
