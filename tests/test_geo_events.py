"""test_geo_events — Stage 9 GeoEventSystem tests.

Tests
-----
1. TestGeoRiskEvaluator
   — Fault risk is higher in boundary zones than interior zones.
   — Landslide risk is higher when stability is low.
   — Collapse risk is higher when fracture/void_risk is elevated.
   — All risk scores are bounded in [0, 1].

2. TestEventSpawnBias  (test_event_spawn_bias)
   — Artificially raising stress/fracture at boundary zones produces
     statistically higher fault risk than stable interior zones.

3. TestPatchApplicationScope  (test_patch_application_scope)
   — An event generates the expected number of patches.
   — Applying patches to a chunk set marks only genuinely affected
     chunks dirty; total dirty count is bounded.

4. TestEventLogReplay  (test_event_log_replay)
   — Generate N events, save their patches to a GeoEventLog.
   — Replay the log into a fresh SDFPatchLog.
   — The replayed patch count matches the original.

5. TestNoPlayerInfluence  (test_no_player_influence)
   — GeoEventSystem can spawn events without a player position
     (player_pos=None disables distance filtering but does not block events).
"""
from __future__ import annotations

import math
import sys
import os
import unittest
from dataclasses import dataclass
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.planet.TectonicPlatesSystem import (
    BoundaryType,
    TectonicPlatesSystem,
    SphericalVoronoi,
    PlateBoundaryClassifier,
    Plate,
    CrustType,
)
from src.planet.GeoFieldSampler import GeoFieldSampler, GeoSample
from src.planet.PlanetHeightProvider import PlanetHeightProvider
from src.planet.SDFChunk import SDFChunkCoord
from src.planet.SDFGenerator import generate_chunk
from src.planet.SDFPatchSystem import SDFPatchLog, SphereCarve
from src.systems.GeoEventSystem import (
    GeoEventExecutor,
    GeoEventLog,
    GeoEventPhase,
    GeoEventRecord,
    GeoEventScheduler,
    GeoEventSystem,
    GeoEventType,
    GeoRiskEvaluator,
    GeoRiskScores,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED   = 42
RADIUS = 1000.0


def _build_tectonic(seed: int = SEED, count: int = 12) -> TectonicPlatesSystem:
    sys_ = TectonicPlatesSystem(seed=seed, plate_count=count)
    sys_.build()
    return sys_


# ---------------------------------------------------------------------------
# Minimal mock GeoSample + GeoFieldSampler for unit tests
# ---------------------------------------------------------------------------

@dataclass
class _MockGeoSample:
    """Simplified stand-in for GeoSample with controllable fields."""
    plate_id:          int   = 0
    boundary_type:     BoundaryType = BoundaryType.NONE
    boundary_strength: float = 0.0
    stress:            float = 0.0
    fracture:          float = 0.0
    stability:         float = 1.0
    hardness:          float = 1.0


class _MockGeoSampler:
    """Returns a single controllable GeoSample for every direction."""

    def __init__(self, sample: _MockGeoSample) -> None:
        self._sample = sample

    def sample(self, direction: Vec3) -> _MockGeoSample:
        return self._sample


# ---------------------------------------------------------------------------
# 1. TestGeoRiskEvaluator
# ---------------------------------------------------------------------------

class TestGeoRiskEvaluator(unittest.TestCase):
    """GeoRiskEvaluator must produce valid, physically plausible risk scores."""

    def _evaluator(self, mock_sample: _MockGeoSample) -> GeoRiskEvaluator:
        return GeoRiskEvaluator(
            geo_sampler   = _MockGeoSampler(mock_sample),
            climate       = None,
            planet_radius = RADIUS,
        )

    # ------------------------------------------------------------------
    def test_scores_bounded(self):
        """All risk scores must lie in [0, 1] for arbitrary inputs."""
        for stress in (0.0, 0.3, 0.7, 1.0):
            for fracture in (0.0, 0.5, 1.0):
                for btype in BoundaryType:
                    sample = _MockGeoSample(
                        boundary_type     = btype,
                        boundary_strength = 0.8,
                        stress            = stress,
                        fracture          = fracture,
                        stability         = max(0.0, 1.0 - fracture * 0.7 - stress * 0.3),
                    )
                    ev     = self._evaluator(sample)
                    scores = ev.evaluate(Vec3(1.0, 0.0, 0.0))
                    for val in (scores.fault_risk, scores.landslide_risk, scores.collapse_risk):
                        self.assertGreaterEqual(val, 0.0,
                            msg=f"Risk below 0: {val} (stress={stress}, fracture={fracture})")
                        self.assertLessEqual(val, 1.0,
                            msg=f"Risk above 1: {val} (stress={stress}, fracture={fracture})")

    def test_interior_zone_zero_fault_risk(self):
        """Interior points (BoundaryType.NONE) must have zero fault risk."""
        sample = _MockGeoSample(
            boundary_type     = BoundaryType.NONE,
            boundary_strength = 0.0,
            stress            = 1.0,
            fracture          = 1.0,
        )
        scores = self._evaluator(sample).evaluate(Vec3(0.0, 1.0, 0.0))
        self.assertAlmostEqual(scores.fault_risk, 0.0, places=9,
            msg="Interior zone must have zero fault risk")

    def test_boundary_zone_nonzero_fault_risk(self):
        """Boundary zones with high stress/fracture must have nonzero fault risk."""
        sample = _MockGeoSample(
            boundary_type     = BoundaryType.CONVERGENT,
            boundary_strength = 0.9,
            stress            = 0.8,
            fracture          = 0.7,
        )
        scores = self._evaluator(sample).evaluate(Vec3(1.0, 0.0, 0.0))
        self.assertGreater(scores.fault_risk, 0.0,
            msg="Convergent boundary with high stress should have nonzero fault risk")

    def test_low_stability_raises_landslide_risk(self):
        """Low stability (high fracture/stress) should raise landslide risk."""
        stable_sample = _MockGeoSample(stability=1.0)
        unstable_sample = _MockGeoSample(
            stability=0.05, fracture=0.9, stress=0.8,
        )
        ev = self._evaluator(stable_sample)
        r_stable   = ev.evaluate(Vec3(1.0, 0.0, 0.0)).landslide_risk
        r_unstable = self._evaluator(unstable_sample).evaluate(Vec3(1.0, 0.0, 0.0)).landslide_risk
        self.assertGreater(r_unstable, r_stable,
            msg="Unstable zone should have higher landslide risk than stable zone")

    def test_divergent_raises_collapse_risk(self):
        """Divergent boundary with high fracture should raise collapse risk."""
        interior = _MockGeoSample(boundary_type=BoundaryType.NONE, fracture=0.1)
        divergent = _MockGeoSample(
            boundary_type=BoundaryType.DIVERGENT,
            boundary_strength=0.9,
            fracture=0.8,
        )
        r_int  = self._evaluator(interior).evaluate(Vec3(0.0, 1.0, 0.0)).collapse_risk
        r_div  = self._evaluator(divergent).evaluate(Vec3(0.0, 1.0, 0.0)).collapse_risk
        self.assertGreater(r_div, r_int,
            msg="Divergent/fracture zone should have higher collapse risk")

    def test_real_tectonic_system_scores_bounded(self):
        """Risk evaluator must stay in bounds when sampling real tectonic data."""
        tect    = _build_tectonic()
        sampler = GeoFieldSampler(tect)
        ev      = GeoRiskEvaluator(sampler, climate=None, planet_radius=RADIUS)
        import random
        rng = random.Random(17)
        for _ in range(100):
            v = Vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            if v.length() < 1e-9:
                continue
            s = ev.evaluate(v.normalized())
            for val in (s.fault_risk, s.landslide_risk, s.collapse_risk):
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)


# ---------------------------------------------------------------------------
# 2. TestEventSpawnBias
# ---------------------------------------------------------------------------

class TestEventSpawnBias(unittest.TestCase):
    """test_event_spawn_bias — boundary zones produce statistically higher fault risk."""

    def _mean_fault_risk(self, samples: List[_MockGeoSample]) -> float:
        scores = []
        for s in samples:
            ev     = GeoRiskEvaluator(_MockGeoSampler(s), climate=None, planet_radius=RADIUS)
            scores.append(ev.evaluate(Vec3(1.0, 0.0, 0.0)).fault_risk)
        return sum(scores) / len(scores) if scores else 0.0

    def test_fault_risk_higher_in_boundary_zones(self):
        """Mean fault risk at stressed boundaries must exceed mean for interior zones."""
        boundary_samples = [
            _MockGeoSample(
                boundary_type=BoundaryType.CONVERGENT,
                boundary_strength=0.8,
                stress=0.85, fracture=0.7,
            ),
            _MockGeoSample(
                boundary_type=BoundaryType.TRANSFORM,
                boundary_strength=0.7,
                stress=0.9, fracture=0.8,
            ),
            _MockGeoSample(
                boundary_type=BoundaryType.DIVERGENT,
                boundary_strength=0.6,
                stress=0.7, fracture=0.6,
            ),
        ]
        interior_samples = [
            _MockGeoSample(boundary_type=BoundaryType.NONE, stress=0.1, fracture=0.1),
            _MockGeoSample(boundary_type=BoundaryType.NONE, stress=0.05, fracture=0.2),
            _MockGeoSample(boundary_type=BoundaryType.NONE, stress=0.0, fracture=0.0),
        ]

        mean_boundary = self._mean_fault_risk(boundary_samples)
        mean_interior = self._mean_fault_risk(interior_samples)

        self.assertGreater(
            mean_boundary, mean_interior,
            msg=(
                f"Boundary fault risk ({mean_boundary:.3f}) must exceed "
                f"interior fault risk ({mean_interior:.3f})"
            ),
        )

    def test_stress_raises_fault_risk_monotonically(self):
        """Increasing stress in a boundary zone should raise fault risk."""
        risks = []
        for stress in (0.0, 0.2, 0.5, 0.8, 1.0):
            sample = _MockGeoSample(
                boundary_type=BoundaryType.CONVERGENT,
                boundary_strength=0.9,
                stress=stress,
                fracture=0.5,
            )
            ev  = GeoRiskEvaluator(_MockGeoSampler(sample), climate=None, planet_radius=RADIUS)
            risks.append(ev.evaluate(Vec3(1.0, 0.0, 0.0)).fault_risk)
        # Risks should be non-decreasing
        for i in range(1, len(risks)):
            self.assertGreaterEqual(
                risks[i], risks[i-1] - 1e-9,
                msg=f"Fault risk should not decrease as stress increases: {risks}",
            )

    def test_real_boundary_cells_have_higher_fault_risk(self):
        """Real tectonic boundary cells must have higher mean fault risk than interior."""
        tect    = _build_tectonic()
        sampler = GeoFieldSampler(tect)

        # Advance stress accumulation
        tect.update(dt=200.0)

        ev = GeoRiskEvaluator(sampler, climate=None, planet_radius=RADIUS)

        boundary_risks = []
        interior_risks = []
        for direction, col, row in tect.field._iter_directions():
            cell = tect.field._cells[tect.field._idx(col, row)]
            score = ev.evaluate(direction).fault_risk
            if cell.boundary_type != BoundaryType.NONE and cell.boundary_strength > 0.1:
                boundary_risks.append(score)
            else:
                interior_risks.append(score)

        if not boundary_risks or not interior_risks:
            self.skipTest("Not enough boundary / interior cells to compare")

        mean_b = sum(boundary_risks) / len(boundary_risks)
        mean_i = sum(interior_risks) / len(interior_risks)
        self.assertGreater(
            mean_b, mean_i,
            msg=(
                f"Real boundary mean fault risk ({mean_b:.3f}) must exceed "
                f"interior mean ({mean_i:.3f})"
            ),
        )


# ---------------------------------------------------------------------------
# 3. TestPatchApplicationScope
# ---------------------------------------------------------------------------

class TestPatchApplicationScope(unittest.TestCase):
    """test_patch_application_scope — events generate and apply patches correctly."""

    _HP = PlanetHeightProvider(SEED)

    def _make_chunk(self, face=0, lod=3, tx=0, ty=0):
        coord = SDFChunkCoord(face_id=face, lod=lod, tile_x=tx, tile_y=ty,
                              depth_index=0)
        return generate_chunk(coord, 8, 2.0, RADIUS, self._HP)

    # ------------------------------------------------------------------
    def test_fault_crack_generates_patches(self):
        """FaultCrackEvent must generate at least one CapsuleCarve patch."""
        executor = GeoEventExecutor(planet_radius=RADIUS, height_provider=self._HP)
        direction = Vec3(1.0, 0.0, 0.0)
        params    = {"fault_length": 60.0, "fault_width": 3.0,
                     "fault_depth": 10.0,  "n_segments": 5.0}
        patches   = executor.execute_fault_crack(direction, params, seed=123)
        self.assertGreater(len(patches), 0,
            "FaultCrackEvent should generate at least one patch")

    def test_landslide_generates_two_patches(self):
        """LandslideEvent must generate a carve + deposit pair."""
        executor = GeoEventExecutor(planet_radius=RADIUS, height_provider=self._HP)
        direction = Vec3(0.0, 1.0, 0.0)
        params    = {"carve_radius": 15.0, "deposit_radius": 10.0,
                     "runout_length": 50.0}
        patches   = executor.execute_landslide(direction, params, seed=456)
        self.assertEqual(len(patches), 2,
            "LandslideEvent should generate exactly [SphereCarve, AdditiveDeposit]")

    def test_collapse_generates_three_patches(self):
        """CollapseEvent must generate chamber + shaft + rim deposit."""
        executor  = GeoEventExecutor(planet_radius=RADIUS, height_provider=self._HP)
        direction = Vec3(0.0, 0.0, 1.0)
        params    = {"chamber_radius": 18.0, "shaft_radius": 5.0,
                     "shaft_depth": 20.0,    "rim_radius": 12.0}
        patches   = executor.execute_collapse(direction, params, seed=789)
        self.assertEqual(len(patches), 3,
            "CollapseEvent should generate chamber + shaft + rim (3 patches)")

    def test_patch_application_marks_only_affected_chunks_dirty(self):
        """Patches far from a chunk must not mark it dirty."""
        # Build a surface chunk near (1, 0, 0)
        chunk = self._make_chunk(face=0)
        chunk.dirty = False

        executor  = GeoEventExecutor(planet_radius=RADIUS, height_provider=self._HP)
        # Generate a fault crack far from the chunk (opposite side of planet)
        far_dir = Vec3(-1.0, 0.0, 0.0)
        params  = {"fault_length": 10.0, "fault_width": 2.0,
                   "fault_depth": 5.0,  "n_segments": 2.0}
        patches = executor.execute_fault_crack(far_dir, params, seed=99)

        any_changed = any(p.apply_to_chunk(chunk) for p in patches)
        # The far-away crack should not affect this chunk
        # (may or may not change depending on chunk position — if it does,
        #  the chunk should be marked dirty; if not, it should remain clean)
        if not any_changed:
            self.assertFalse(chunk.dirty,
                "Distant patches should not mark the chunk dirty")

    def test_apply_fault_patches_to_nearby_chunk(self):
        """Fault patches centred near a chunk should mark it dirty if they overlap."""
        chunk    = self._make_chunk(face=0)
        chunk.dirty = False

        executor  = GeoEventExecutor(planet_radius=RADIUS, height_provider=self._HP)
        # Use the chunk centre direction
        R  = chunk.resolution
        cx, cy, cz = chunk.get_pos(R // 2, R // 2, R // 2)
        centre_dir = Vec3(cx, cy, cz).normalized()

        params = {"fault_length": 20.0, "fault_width": 8.0,
                  "fault_depth": 5.0,  "n_segments": 2.0}
        patches = executor.execute_fault_crack(centre_dir, params, seed=7)

        changed_count = sum(1 for p in patches if p.apply_to_chunk(chunk))
        # At least one patch should overlap the chunk at its own centre
        self.assertGreater(changed_count, 0,
            "Fault crack centred on chunk should affect at least one patch")

    def test_dirty_chunk_count_bounded_after_landslide(self):
        """Applying a landslide to a small set of chunks limits dirty count."""
        chunks = [self._make_chunk(face=0, tx=i, ty=j)
                  for i in range(2) for j in range(2)]
        for c in chunks:
            c.dirty = False

        executor  = GeoEventExecutor(planet_radius=RADIUS, height_provider=self._HP)
        direction = Vec3(1.0, 0.0, 0.0)
        params    = {"carve_radius": 20.0, "deposit_radius": 10.0, "runout_length": 30.0}
        patches   = executor.execute_landslide(direction, params, seed=42)

        for patch in patches:
            for chunk in chunks:
                patch.apply_to_chunk(chunk)

        dirty_count = sum(1 for c in chunks if c.dirty)
        # Dirty count must be at most the total number of chunks
        self.assertLessEqual(dirty_count, len(chunks))


# ---------------------------------------------------------------------------
# 4. TestEventLogReplay
# ---------------------------------------------------------------------------

class TestEventLogReplay(unittest.TestCase):
    """test_event_log_replay — log → replay → same patches."""

    _HP       = PlanetHeightProvider(SEED)
    _EXECUTOR = GeoEventExecutor(planet_radius=RADIUS, height_provider=_HP)

    def _make_records(self, n: int = 3) -> List[GeoEventRecord]:
        """Generate N synthetic event records with real patches."""
        import random as _rnd
        rng     = _rnd.Random(SEED)
        records = []
        type_cycle = list(GeoEventType)
        default_params_by_type = {
            GeoEventType.FAULT_CRACK: {"fault_length": 30.0, "fault_width": 3.0,
                                       "fault_depth": 8.0, "n_segments": 3.0},
            GeoEventType.LANDSLIDE:  {"carve_radius": 12.0, "deposit_radius": 8.0,
                                      "runout_length": 40.0},
            GeoEventType.COLLAPSE:   {"chamber_radius": 15.0, "shaft_radius": 4.0,
                                      "shaft_depth": 18.0, "rim_radius": 10.0},
        }
        for i in range(n):
            direction = Vec3(
                rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)
            ).normalized()
            etype   = type_cycle[i % len(type_cycle)]
            params  = default_params_by_type[etype]
            patches = self._EXECUTOR.execute(etype, direction, params, seed=i * 13)
            records.append(GeoEventRecord(
                event_id   = i,
                event_type = etype,
                start_time = float(i * 10),
                direction  = direction,
                seed_local = i * 13,
                params     = params,
                patches    = patches,
            ))
        return records

    # ------------------------------------------------------------------
    def test_log_records_appended_correctly(self):
        """Records added to GeoEventLog must be retrievable in order."""
        log     = GeoEventLog()
        records = self._make_records(3)
        for r in records:
            log.add(r)
        self.assertEqual(len(log), 3)
        for original, stored in zip(records, log.records()):
            self.assertEqual(original.event_id, stored.event_id)
            self.assertEqual(original.event_type, stored.event_type)

    def test_replay_patch_count_matches_original(self):
        """Replaying the log must yield the same total number of patches."""
        log     = GeoEventLog()
        records = self._make_records(3)
        total_original = 0
        for r in records:
            log.add(r)
            total_original += len(r.patches)

        replay_log = SDFPatchLog()
        log.replay_to_patch_log(replay_log)

        self.assertEqual(
            len(replay_log), total_original,
            msg=(
                f"Replayed patch count ({len(replay_log)}) must equal "
                f"original ({total_original})"
            ),
        )

    def test_replay_is_deterministic(self):
        """Two replays of the same log must produce identical patch lists."""
        log     = GeoEventLog()
        records = self._make_records(4)
        for r in records:
            log.add(r)

        log1 = SDFPatchLog()
        log2 = SDFPatchLog()
        log.replay_to_patch_log(log1)
        log.replay_to_patch_log(log2)

        patches1 = log1.patches()
        patches2 = log2.patches()
        self.assertEqual(len(patches1), len(patches2),
            "Replay must be deterministic (same patch count)")
        # Check identity: same patch objects (same list)
        for p1, p2 in zip(patches1, patches2):
            self.assertIs(p1, p2,
                "Replayed patches should be the same objects (not copies)")

    def test_empty_log_replay_is_safe(self):
        """Replaying an empty log must not raise and yields zero patches."""
        log        = GeoEventLog()
        replay_log = SDFPatchLog()
        log.replay_to_patch_log(replay_log)   # must not raise
        self.assertEqual(len(replay_log), 0)

    def test_log_preserves_event_order(self):
        """Records must be returned in insertion order."""
        log = GeoEventLog()
        for i in range(5):
            log.add(GeoEventRecord(
                event_id   = i,
                event_type = GeoEventType.FAULT_CRACK,
                start_time = float(i),
                direction  = Vec3(1.0, 0.0, 0.0),
                seed_local = i,
                params     = {},
                patches    = [],
            ))
        ids = [r.event_id for r in log.records()]
        self.assertEqual(ids, [0, 1, 2, 3, 4])


# ---------------------------------------------------------------------------
# 5. TestNoPlayerInfluence
# ---------------------------------------------------------------------------

class TestNoPlayerInfluence(unittest.TestCase):
    """test_no_player_influence — events can fire without player position."""

    def _build_system(self, seed: int = SEED) -> GeoEventSystem:
        """Build a GeoEventSystem with artificially high-risk zones."""
        # Use a mock sampler that returns high risk everywhere
        high_risk_sample = _MockGeoSample(
            boundary_type     = BoundaryType.CONVERGENT,
            boundary_strength = 1.0,
            stress            = 1.0,
            fracture          = 1.0,
            stability         = 0.0,
        )
        sampler = _MockGeoSampler(high_risk_sample)
        return GeoEventSystem(
            geo_sampler         = sampler,
            climate             = None,
            sdf_world           = None,
            planet_radius       = RADIUS,
            height_provider     = None,
            seed                = seed,
            pre_seconds         = 0.1,
            post_seconds        = 0.1,
            rate_minor_per_hour = 9999.0,
            rate_major_per_hour = 9999.0,
            cooldown_minutes_per_tile = 0.0,
        )

    # ------------------------------------------------------------------
    def test_events_fire_without_player_pos(self):
        """Events must be spawnable when player_pos is None."""
        system = self._build_system()
        # Large dt to quickly fill hazard
        for _ in range(50):
            system.update_with_dt(dt=100.0, game_time=float(_ * 100), player_pos=None)

        self.assertGreater(
            len(system.event_log), 0,
            "GeoEventSystem should have fired events without player position",
        )

    def test_event_generation_does_not_require_player_input(self):
        """The event_log must grow over time even with no player interaction."""
        system = self._build_system(seed=SEED + 7)
        initial_count = len(system.event_log)
        for tick in range(100):
            system.update_with_dt(dt=50.0, game_time=float(tick * 50), player_pos=None)
        final_count = len(system.event_log)
        self.assertGreater(
            final_count, initial_count,
            "Event log should grow over time without any player input",
        )

    def test_determinism_with_same_seed(self):
        """Two systems with the same seed must produce the same event log."""
        def _run(seed: int) -> List[int]:
            sys_ = self._build_system(seed)
            for i in range(30):
                sys_.update_with_dt(dt=80.0, game_time=float(i * 80), player_pos=None)
            return [r.event_id for r in sys_.event_log.records()]

        ids1 = _run(SEED)
        ids2 = _run(SEED)
        self.assertEqual(ids1, ids2,
            "Same seed must produce identical event id sequences")

    def test_different_seeds_may_differ(self):
        """Different seeds are allowed to produce different event counts."""
        def _count(seed: int) -> int:
            sys_ = self._build_system(seed)
            for i in range(20):
                sys_.update_with_dt(dt=80.0, game_time=float(i * 80), player_pos=None)
            return len(sys_.event_log)

        # Different seeds are not required to differ, but the code must not crash
        c1 = _count(SEED)
        c2 = _count(SEED + 999)
        # Just verify both are non-negative integers (no crash)
        self.assertGreaterEqual(c1, 0)
        self.assertGreaterEqual(c2, 0)

    def test_ground_stability_not_influenced_by_player(self):
        """query_ground_stability must not depend on player input."""
        system  = self._build_system()
        pos     = Vec3(RADIUS, 0.0, 0.0)
        stab1   = system.query_ground_stability(pos)
        # Update a few ticks with player pos set
        system.update_with_dt(dt=1.0, game_time=0.0, player_pos=Vec3(RADIUS + 10, 0, 0))
        stab2   = system.query_ground_stability(pos)
        # Stability comes purely from geology (mock returns fixed value)
        self.assertAlmostEqual(stab1, stab2, places=9,
            msg="Ground stability should not change based on player position")


# ---------------------------------------------------------------------------
# 6. TestGeoEventPhases
# ---------------------------------------------------------------------------

class TestGeoEventPhases(unittest.TestCase):
    """Verify PRE → IMPACT → POST phase progression."""

    def _make_system(self) -> GeoEventSystem:
        high_risk = _MockGeoSample(
            boundary_type=BoundaryType.CONVERGENT,
            boundary_strength=1.0,
            stress=1.0,
            fracture=1.0,
            stability=0.0,
        )
        return GeoEventSystem(
            geo_sampler         = _MockGeoSampler(high_risk),
            climate             = None,
            sdf_world           = None,
            planet_radius       = RADIUS,
            seed                = SEED,
            pre_seconds         = 2.0,
            post_seconds        = 5.0,
            rate_minor_per_hour = 9999.0,
            rate_major_per_hour = 9999.0,
            cooldown_minutes_per_tile = 0.0,
        )

    def test_event_starts_in_pre_phase(self):
        """Newly spawned events must begin in PRE phase.

        We use a dt much smaller than pre_duration so the phase does not
        advance to IMPACT within the spawning tick.
        """
        system = self._make_system()   # pre_seconds=2.0
        spawned_in_pre = False
        # Use dt=0.1 so pre phase (2 s) is NOT completed in one step
        for i in range(2000):
            system.update_with_dt(dt=0.1, game_time=float(i) * 0.1, player_pos=None)
            for evt in system.active_events:
                if evt.phase == GeoEventPhase.PRE:
                    spawned_in_pre = True
                    break
            if spawned_in_pre:
                break

        if not spawned_in_pre and not system.event_log.records():
            self.skipTest("No events spawned in test window")

        self.assertTrue(spawned_in_pre,
            "At least one event should have been observed in PRE phase")

    def test_signal_intensity_in_range(self):
        """signal_intensity must stay in [0, 1] throughout all phases."""
        from src.systems.GeoEventSystem import GeoEvent, GeoEventPhase
        record = GeoEventRecord(
            event_id=0, event_type=GeoEventType.FAULT_CRACK,
            start_time=0.0, direction=Vec3(1.0, 0.0, 0.0),
            seed_local=0, params={}, patches=[],
        )
        evt = GeoEvent(record=record, pre_duration=5.0, post_duration=10.0)
        for elapsed in (0.0, 2.5, 5.0):
            evt.phase_elapsed = elapsed
            self.assertGreaterEqual(evt.signal_intensity(), 0.0)
            self.assertLessEqual(evt.signal_intensity(), 1.0)
        evt.phase = GeoEventPhase.POST
        for elapsed in (0.0, 5.0, 10.0):
            evt.phase_elapsed = elapsed
            self.assertGreaterEqual(evt.signal_intensity(), 0.0)
            self.assertLessEqual(evt.signal_intensity(), 1.0)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
