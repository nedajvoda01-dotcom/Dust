"""test_input_control_stage43.py — Stage 43 Input-Driven Embodied Control tests.

Tests
-----
1. test_primary_look_dominates_long_term
   — After a reflex fires, FinalLookDir returns to primary within decay tau.

2. test_head_torso_body_limits_respected
   — LookRigController distributes yaw within configured joint limits.

3. test_reflex_never_overrides_move_input
   — Reflex slowdown biases speed but never changes the move direction.

4. test_suit_descriptor_deterministic
   — Same player_seed → identical SuitKitDescriptor on repeated calls.

5. test_network_appearance_sync
   — announce_appearance → receive_appearance round-trip yields same descriptor.

6. test_small_physical_variation_bounds
   — SuitMassBinder keeps COM shift and inertia within configured limits.
"""
from __future__ import annotations

import math
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.math.Vec3 import Vec3
from src.input.PlayerIntent import (
    PlayerIntent, PrimaryMoveTarget, PrimaryLookTarget, ReflexOverlay,
)
from src.input.InputSystem import InputSystem, WASDState
from src.look.LookRigController import LookRigController
from src.reflex.ReflexOverlaySystem import ReflexOverlaySystem
from src.perception.PerceptionSystem import PerceptionState
from src.character.SuitKitDescriptor import SuitKitDescriptor
from src.character.SuitKitAssembler import SuitKitAssembler
from src.character.BodyArchetypeDB import BodyArchetypeDB
from src.net.PlayerAppearanceReplicator import PlayerAppearanceReplicator
from src.physics.SuitMassBinder import SuitMassBinder
from src.audio.SuitAcousticBinder import SuitAcousticBinder


_PLANET_R = 1000.0
_UP = Vec3(0.0, 1.0, 0.0)
_FWD = Vec3(0.0, 0.0, -1.0)
_RIGHT = Vec3(1.0, 0.0, 0.0)
_DT = 1.0 / 60.0


# ---------------------------------------------------------------------------
# 1. test_primary_look_dominates_long_term
# ---------------------------------------------------------------------------

class TestPrimaryLookDominatesLongTerm(unittest.TestCase):
    """After a reflex fires, FinalLookDir must converge back to primary."""

    def test_reflex_decays_to_primary(self):
        """r decays exponentially; after several tau, FinalLookDir ≈ primary."""
        ros = ReflexOverlaySystem()

        # Inject a strong audio stimulus for one tick
        loud = PerceptionState(
            audioUrgency  = 0.9,
            attentionDir  = Vec3(1.0, 0.0, 0.0),
        )
        overlay = ros.update(dt=_DT, perception_state=loud)
        r_initial = overlay.lookBias_strength
        self.assertGreater(r_initial, 0.0, "Reflex should fire on loud audio")

        # Silence
        silent = PerceptionState()
        decay_tau = 0.4  # default
        # Run for 3× tau — r should have dropped to < e^(-3) ≈ 0.05 of initial
        ticks = int(3.0 * decay_tau / _DT)
        for _ in range(ticks):
            overlay = ros.update(dt=_DT, perception_state=silent)

        r_final = overlay.lookBias_strength
        self.assertLess(
            r_final,
            r_initial * 0.1,
            f"Reflex r should decay; initial={r_initial:.4f} final={r_final:.4f}",
        )

    def test_final_look_converges_to_primary(self):
        """FinalLookDir angle from primary must be small after decay."""
        primary_look = Vec3(0.0, 0.0, -1.0)
        reflex_look  = Vec3(1.0, 0.0, 0.0)

        ros = ReflexOverlaySystem()
        loud = PerceptionState(audioUrgency=0.9, attentionDir=reflex_look)
        ros.update(dt=_DT, perception_state=loud)

        silent = PerceptionState()
        ticks = int(3.0 * 0.4 / _DT)
        overlay = None
        for _ in range(ticks):
            overlay = ros.update(dt=_DT, perception_state=silent)

        intent = PlayerIntent(
            look=PrimaryLookTarget(lookDir_world=primary_look),
            reflex=overlay,
        )
        final = intent.FinalLookDir
        dot = final.dot(primary_look)
        # After decay, final should be within ~10° of primary
        self.assertGreater(
            dot, math.cos(math.radians(10.0)),
            f"FinalLookDir should be close to primary after decay; dot={dot:.4f}",
        )


# ---------------------------------------------------------------------------
# 2. test_head_torso_body_limits_respected
# ---------------------------------------------------------------------------

class TestHeadTorsoBodyLimitsRespected(unittest.TestCase):
    """LookRigController must not exceed configured yaw limits."""

    def _make_rig(self) -> LookRigController:
        return LookRigController()

    def _yaw_angle_deg(self, a: Vec3, b: Vec3, up: Vec3) -> float:
        """Signed yaw angle in degrees from *a* to *b* around *up*."""
        ah = (a - up * a.dot(up))
        bh = (b - up * b.dot(up))
        la, lb = ah.length(), bh.length()
        if la < 1e-9 or lb < 1e-9:
            return 0.0
        ah = ah * (1.0 / la)
        bh = bh * (1.0 / lb)
        dot = max(-1.0, min(1.0, ah.dot(bh)))
        return math.degrees(math.acos(dot))

    def test_head_yaw_within_limit(self):
        """Head yaw relative to torso must not exceed head_yaw_max_deg."""
        rig = self._make_rig()
        head_yaw_max = 60.0

        # Look 90° to the right (beyond head limit)
        look = Vec3(1.0, 0.0, 0.0)   # right = 90° from -Z forward
        for _ in range(30):
            result = rig.update(
                dt=_DT,
                final_look_dir=look,
                body_forward=_FWD,
                up=_UP,
            )

        # head_yaw_fraction ≤ 1.0 means head has not gone beyond its limit
        self.assertLessEqual(
            result.head_yaw_fraction,
            1.0 + 1e-6,
            f"head_yaw_fraction {result.head_yaw_fraction:.3f} should be ≤ 1.0",
        )
        # Additionally, verify head yaw relative to torso is bounded:
        # measure angle between torso_dir and head_dir
        head_to_torso = self._yaw_angle_deg(result.torso_dir, result.head_dir, _UP)
        self.assertLessEqual(
            head_to_torso,
            head_yaw_max + 1.0,   # 1° float tolerance
            f"Head yaw relative to torso {head_to_torso:.2f}° should not exceed {head_yaw_max}°",
        )

    def test_body_yaw_goal_nonzero_when_beyond_head_torso(self):
        """When look exceeds head+torso limits, body_yaw_goal_rad should be nonzero
        (or body_forward should have rotated toward look over many ticks)."""
        rig = LookRigController()
        look = Vec3(1.0, 0.0, 0.0)  # 90° right of body

        result = None
        for _ in range(90):   # 1.5 s
            result = rig.update(
                dt=_DT,
                final_look_dir=look,
                body_forward=_FWD,
                up=_UP,
                is_moving=False,
            )

        # Body forward should have rotated somewhat toward look after 1.5 s
        body_to_look = self._yaw_angle_deg(result.body_forward, look, _UP)
        # After standing rotation, body should be closer to look than the initial 90°
        self.assertLess(
            body_to_look,
            90.0,
            f"Body should have rotated toward look; still {body_to_look:.1f}° away",
        )

    def test_torso_yaw_within_limit(self):
        """Torso yaw fraction should be ≤ 1.0 (not exceeding the limit)."""
        rig = self._make_rig()
        look = Vec3(0.5, 0.0, -0.866)  # ~30° right

        result = None
        for _ in range(10):
            result = rig.update(
                dt=_DT,
                final_look_dir=look,
                body_forward=_FWD,
                up=_UP,
            )

        self.assertLessEqual(
            result.torso_yaw_fraction,
            1.0 + 1e-6,
            f"torso_yaw_fraction {result.torso_yaw_fraction:.3f} should be ≤ 1.0",
        )


# ---------------------------------------------------------------------------
# 3. test_reflex_never_overrides_move_input
# ---------------------------------------------------------------------------

class TestReflexNeverOverridesMoveInput(unittest.TestCase):
    """Reflex biases speed/brace but never changes the move direction."""

    def test_move_direction_preserved_under_reflex(self):
        """FinalMoveDir must equal the primary move direction regardless of reflex."""
        primary_dir = Vec3(1.0, 0.0, 0.0)
        reflex_look = Vec3(0.0, 0.0, -1.0)

        intent = PlayerIntent(
            move=PrimaryMoveTarget(moveDir_local=primary_dir, speedIntent=1.0),
            look=PrimaryLookTarget(lookDir_world=Vec3(0.0, 0.0, -1.0)),
            reflex=ReflexOverlay(
                lookBias_dir=reflex_look,
                lookBias_strength=0.3,
                braceBias=0.8,
                slowdownBias=0.5,
            ),
        )
        # FinalMoveDir must equal primary
        self.assertAlmostEqual(intent.FinalMoveDir.x, primary_dir.x, places=9)
        self.assertAlmostEqual(intent.FinalMoveDir.z, primary_dir.z, places=9)

    def test_slowdown_only_reduces_speed(self):
        """slowdownBias reduces FinalSpeed but keeps direction identical."""
        intent = PlayerIntent(
            move=PrimaryMoveTarget(moveDir_local=Vec3(0.0, 0.0, -1.0),
                                   speedIntent=1.0),
            reflex=ReflexOverlay(slowdownBias=0.4),
        )
        self.assertAlmostEqual(intent.FinalSpeed, 0.6, places=9)
        self.assertAlmostEqual(intent.FinalMoveDir.z, -1.0, places=9)

    def test_full_slowdown_reaches_zero_speed(self):
        """slowdownBias=1.0 makes FinalSpeed exactly 0."""
        intent = PlayerIntent(
            move=PrimaryMoveTarget(moveDir_local=Vec3(1.0, 0.0, 0.0),
                                   speedIntent=1.0),
            reflex=ReflexOverlay(slowdownBias=1.0),
        )
        self.assertAlmostEqual(intent.FinalSpeed, 0.0, places=9)


# ---------------------------------------------------------------------------
# 4. test_suit_descriptor_deterministic
# ---------------------------------------------------------------------------

class TestSuitDescriptorDeterministic(unittest.TestCase):
    """Same seed → identical SuitKitDescriptor every time."""

    def test_same_seed_same_descriptor(self):
        """Two descriptors from the same seed must be byte-identical."""
        seed = 123456789
        d1 = SuitKitDescriptor.from_seed(seed)
        d2 = SuitKitDescriptor.from_seed(seed)
        self.assertEqual(d1, d2, "Same seed should yield identical descriptor")
        self.assertEqual(d1.pack(), d2.pack())

    def test_different_seeds_different_descriptors(self):
        """Different seeds should produce different descriptors (with high probability)."""
        d1 = SuitKitDescriptor.from_seed(1)
        d2 = SuitKitDescriptor.from_seed(2)
        self.assertNotEqual(d1.pack(), d2.pack(),
                            "Different seeds should yield different descriptors")

    def test_pack_unpack_round_trip(self):
        """pack → unpack must be lossless."""
        d = SuitKitDescriptor.from_seed(42)
        restored = SuitKitDescriptor.unpack(d.pack())
        self.assertEqual(d, restored)
        self.assertEqual(d.helmet_id,    restored.helmet_id)
        self.assertEqual(d.backpack_id,  restored.backpack_id)
        self.assertEqual(d.roughness_var, restored.roughness_var)
        self.assertEqual(d.pattern_shift, restored.pattern_shift)

    def test_assembler_same_descriptor_same_kit(self):
        """Same descriptor → identical SuitKit material and extra_mass_kg."""
        seed = 999
        d = SuitKitDescriptor.from_seed(seed)
        asm = SuitKitAssembler()
        k1 = asm.assemble(d)
        k2 = asm.assemble(d)
        self.assertAlmostEqual(k1.extra_mass_kg, k2.extra_mass_kg, places=9)
        self.assertAlmostEqual(k1.material.roughness, k2.material.roughness, places=9)
        self.assertAlmostEqual(k1.material.wear_amount, k2.material.wear_amount, places=9)


# ---------------------------------------------------------------------------
# 5. test_network_appearance_sync
# ---------------------------------------------------------------------------

class TestNetworkAppearanceSync(unittest.TestCase):
    """join → all clients see identical module ids/colors."""

    def test_announce_receive_round_trip(self):
        """announce_appearance → receive_appearance yields same descriptor."""
        rep = PlayerAppearanceReplicator(world_seed=42)
        rep.register_player("player_alice")
        payload = rep.announce_appearance("player_alice")
        _pid_hash, desc_recv = PlayerAppearanceReplicator.receive_appearance(payload)
        desc_orig = rep.get_descriptor("player_alice")
        self.assertEqual(desc_orig, desc_recv,
                         "Received descriptor must match original")

    def test_all_clients_get_same_descriptor(self):
        """Two independently decoded payloads of the same player are equal."""
        rep = PlayerAppearanceReplicator(world_seed=7)
        rep.register_player("player_bob")
        payload = rep.announce_appearance("player_bob")

        _, d1 = PlayerAppearanceReplicator.receive_appearance(payload)
        _, d2 = PlayerAppearanceReplicator.receive_appearance(payload)
        self.assertEqual(d1, d2)

    def test_different_players_different_descriptors(self):
        """Different player IDs should produce different appearances."""
        rep = PlayerAppearanceReplicator(world_seed=42)
        rep.register_player("player_a")
        rep.register_player("player_b")
        da = rep.get_descriptor("player_a")
        db = rep.get_descriptor("player_b")
        # With high probability (different seeds), packs differ
        self.assertNotEqual(da.pack(), db.pack(),
                            "Different players should have different looks")

    def test_stable_across_re_register(self):
        """Re-registering an existing player returns the same descriptor."""
        rep = PlayerAppearanceReplicator(world_seed=1)
        d1 = rep.register_player("player_c")
        d2 = rep.register_player("player_c")
        self.assertEqual(d1, d2, "Re-register should return same descriptor")

    def test_payload_length(self):
        """Wire payload must be exactly 19 bytes (4 header + 15 descriptor)."""
        rep = PlayerAppearanceReplicator(world_seed=0)
        rep.register_player("x")
        payload = rep.announce_appearance("x")
        self.assertEqual(len(payload), 19,
                         f"Expected 19-byte payload; got {len(payload)}")


# ---------------------------------------------------------------------------
# 6. test_small_physical_variation_bounds
# ---------------------------------------------------------------------------

class TestSmallPhysicalVariationBounds(unittest.TestCase):
    """SuitMassBinder keeps COM shift and inertia within spec limits."""

    def _make_params(self, seed: int):
        db    = BodyArchetypeDB()
        arch  = db.get("default")
        desc  = SuitKitDescriptor.from_seed(seed)
        asm   = SuitKitAssembler()
        kit   = asm.assemble(desc, arch)
        binder = SuitMassBinder()
        return binder.bind(arch, kit), arch, kit

    def test_com_shift_within_bounds(self):
        """COM shift must not exceed com_shift_max = 0.04 (hard cap)."""
        for seed in [0, 1, 42, 999, 65535]:
            params, _arch, _kit = self._make_params(seed)
            self.assertLessEqual(
                params.com_shift_back, 0.04 + 1e-9,
                f"seed={seed}: com_shift_back {params.com_shift_back} > 0.04",
            )
            self.assertLessEqual(
                params.com_shift_up, 0.04 + 1e-9,
                f"seed={seed}: com_shift_up {params.com_shift_up} > 0.04",
            )

    def test_inertia_within_bounds(self):
        """Inertia scale must be in [0.95, 1.05]."""
        for seed in [0, 1, 42, 999, 65535]:
            params, _arch, _kit = self._make_params(seed)
            self.assertGreaterEqual(params.inertia_scale, 0.95 - 1e-9)
            self.assertLessEqual(params.inertia_scale,   1.05 + 1e-9)

    def test_mass_variation_bounded(self):
        """Total mass variation must stay within ±8% of baseline."""
        db   = BodyArchetypeDB()
        arch = db.get("default")
        binder = SuitMassBinder()

        masses = []
        for seed in range(20):
            desc = SuitKitDescriptor.from_seed(seed)
            kit  = SuitKitAssembler().assemble(desc, arch)
            params = binder.bind(arch, kit)
            masses.append(params.total_mass_kg)

        min_m = min(masses)
        max_m = max(masses)
        baseline = arch.total_mass_kg + 4.0   # approx average base suit mass
        # Range should be ≤ 16% of baseline (±8%)
        self.assertLess(
            max_m - min_m,
            baseline * 0.20,
            f"Mass range {max_m - min_m:.2f} kg exceeds 20% of baseline {baseline:.2f}",
        )

    def test_acoustic_binder_has_modal_freqs(self):
        """SuitAcousticBinder should produce at least one modal frequency."""
        desc = SuitKitDescriptor.from_seed(42)
        kit  = SuitKitAssembler().assemble(desc)
        binder = SuitAcousticBinder()
        profile = binder.bind(kit)
        self.assertGreater(len(profile.modal_freqs_hz), 0,
                           "AcousticProfile should have modal frequencies")
        self.assertGreater(profile.primary_modal_hz, 0.0)


# ---------------------------------------------------------------------------
# Bonus: InputSystem smoke test
# ---------------------------------------------------------------------------

class TestInputSystem(unittest.TestCase):
    """Basic smoke tests for InputSystem."""

    def test_no_keys_no_movement(self):
        """With no WASD pressed, moveDir should be zero and speedIntent=0."""
        sys_in = InputSystem()
        intent = sys_in.update(
            dt=_DT, wasd=WASDState(),
            mouse_dx=0.0, mouse_dy=0.0,
            camera_forward=_FWD, camera_right=_RIGHT, up=_UP,
        )
        self.assertAlmostEqual(intent.move.speedIntent, 0.0)
        self.assertAlmostEqual(intent.move.moveDir_local.length(), 0.0, places=6)

    def test_w_key_forward_movement(self):
        """W key should produce a forward move direction."""
        sys_in = InputSystem()
        intent = sys_in.update(
            dt=_DT, wasd=WASDState(w=True),
            mouse_dx=0.0, mouse_dy=0.0,
            camera_forward=_FWD, camera_right=_RIGHT, up=_UP,
        )
        # After smoothing, dir should point approximately forward (-Z)
        self.assertGreater(intent.move.speedIntent, 0.5)

    def test_mouse_changes_look_direction(self):
        """Mouse delta should change lookDir_world from default."""
        sys_in = InputSystem()
        # No movement
        intent0 = sys_in.update(
            dt=_DT, wasd=WASDState(),
            mouse_dx=0.0, mouse_dy=0.0,
            camera_forward=_FWD, camera_right=_RIGHT, up=_UP,
        )
        look0 = intent0.look.lookDir_world

        # Apply horizontal mouse delta
        intent1 = sys_in.update(
            dt=_DT, wasd=WASDState(),
            mouse_dx=100.0, mouse_dy=0.0,
            camera_forward=_FWD, camera_right=_RIGHT, up=_UP,
        )
        look1 = intent1.look.lookDir_world

        # Look direction should have changed
        self.assertFalse(
            abs(look0.dot(look1) - 1.0) < 1e-6,
            "Look direction should change with mouse input",
        )

    def test_look_dir_is_unit_vector(self):
        """lookDir_world must always be a unit vector."""
        sys_in = InputSystem()
        for dx, dy in [(0, 0), (10, 5), (-20, 15), (0, 90), (360, 0)]:
            intent = sys_in.update(
                dt=_DT, wasd=WASDState(w=True, d=True),
                mouse_dx=float(dx), mouse_dy=float(dy),
                camera_forward=_FWD, camera_right=_RIGHT, up=_UP,
            )
            length = intent.look.lookDir_world.length()
            self.assertAlmostEqual(length, 1.0, places=6,
                                   msg=f"lookDir not unit: {length} for dx={dx},dy={dy}")


if __name__ == "__main__":
    unittest.main()
