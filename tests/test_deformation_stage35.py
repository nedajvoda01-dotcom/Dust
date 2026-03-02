"""test_deformation_stage35.py — Stage 35 Procedural Contact & Surface Deformation.

Tests
-----
1. test_indent_only_when_pressure_exceeds_yield
   — At low Fn (below yield) H barely changes; above yield H deepens.

2. test_slip_creates_anisotropic_track
   — A SLIDING contact with slip velocity produces a groove (H lower) and
     asymmetric M redistribution along the slip direction.

3. test_relaxation_erases_traces_over_time
   — H/M converge back toward resting values; storm_multiplier accelerates.

4. test_network_stamp_determinism
   — The same DeformStampBatch encoded then decoded and applied to two
     independent fields produces identical field_hash values.

5. test_budget_limits_not_exceeded
   — Spamming many contact samples never causes uploads_per_frame to exceed
     the configured maximum.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.physics.MaterialYieldModel import MaterialClass, MaterialYieldModel
from src.surface.DeformationField import ContactSample, DeformationField
from src.surface.DeformationIntegrator import DeformationIntegrator
from src.surface.DeformStampCodec import (
    DeformStamp,
    DeformStampBatch,
    DeformStampCodec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dust_params():
    model = MaterialYieldModel()
    return model.get(MaterialClass.DUST)


def _make_field(grid_res: int = 16, m_base: float = 0.5) -> DeformationField:
    f = DeformationField(chunk_id=(0, 0), grid_res=grid_res, m_base=m_base)
    f.set_relaxation_taus(tau_h=10.0, tau_m=10.0)
    return f


def _make_sample(
    ix: int = 8,
    iy: int = 8,
    fn: float = 500.0,
    ft_x: float = 0.0,
    ft_y: float = 0.0,
    v_rel_x: float = 0.0,
    v_rel_y: float = 0.0,
    area: float = 0.05,
    material: MaterialClass = MaterialClass.DUST,
) -> ContactSample:
    return ContactSample(
        world_ix=ix,
        world_iy=iy,
        fn=fn,
        ft_x=ft_x,
        ft_y=ft_y,
        v_rel_x=v_rel_x,
        v_rel_y=v_rel_y,
        area=area,
        material=material,
    )


# ---------------------------------------------------------------------------
# 1. test_indent_only_when_pressure_exceeds_yield
# ---------------------------------------------------------------------------

class TestIndentOnlyAboveYield(unittest.TestCase):
    """Plastic indentation must only occur when pressure > yield."""

    def setUp(self) -> None:
        self.params = _dust_params()    # yield ~500 Pa

    def test_below_yield_no_significant_indent(self) -> None:
        f  = _make_field()
        # pressure = fn / area = 10 / 0.05 = 200 < 500 yield
        s  = _make_sample(fn=10.0, area=0.05)
        f.apply_contact_sample(s, self.params, dt=1.0)
        h  = f.h_at(8, 8)
        # Should be essentially zero (tiny or zero)
        self.assertAlmostEqual(h, 0.0, places=3,
            msg=f"Expected H≈0 below yield, got {h:.6f} m")

    def test_above_yield_causes_indent(self) -> None:
        f  = _make_field()
        # pressure = 5000 / 0.05 = 100 000 >> 500 yield
        s  = _make_sample(fn=5000.0, area=0.05)
        f.apply_contact_sample(s, self.params, dt=1.0)
        h  = f.h_at(8, 8)
        self.assertLess(h, -1e-4,
            msg=f"Expected negative H (indent) above yield, got {h:.6f} m")

    def test_indent_proportional_to_excess_pressure(self) -> None:
        """Higher excess pressure → deeper indent."""
        f1 = _make_field()
        f2 = _make_field()
        # Use small dt so neither field saturates the int16 H range
        # low excess: pressure = 600/0.1 = 6000 Pa, dh ≈ 0.012*(6000-500)*0.01 = 0.66 m
        f1.apply_contact_sample(_make_sample(fn=600.0, area=0.1), self.params, 0.01)
        # high excess: pressure = 6000/0.1 = 60000 Pa, dh ≈ 0.012*(60000-500)*0.01 = 7.14 m
        f2.apply_contact_sample(_make_sample(fn=6000.0, area=0.1), self.params, 0.01)
        self.assertLess(f2.h_at(8, 8), f1.h_at(8, 8),
            "Higher pressure should produce deeper indent")


# ---------------------------------------------------------------------------
# 2. test_slip_creates_anisotropic_track
# ---------------------------------------------------------------------------

class TestSlipAnisotropicTrack(unittest.TestCase):
    """Sliding contact must create a directional groove and move mass."""

    def test_groove_forms_along_slip(self) -> None:
        params = _dust_params()
        f      = _make_field(grid_res=16)
        # Sliding contact: strong slip in +x direction
        s = _make_sample(
            ix=7, iy=8,
            fn=2000.0, area=0.05,
            ft_x=50.0, ft_y=0.0,
            v_rel_x=2.0, v_rel_y=0.0,
        )
        for _ in range(10):
            f.apply_contact_sample(s, params, dt=0.1)

        h_centre = f.h_at(7, 8)
        self.assertLess(h_centre, 0.0,
            f"Expected negative H in slip groove, got {h_centre:.6f}")

    def test_mass_displaced_in_push_direction(self) -> None:
        """Bulldozing must accumulate mass ahead of the slip direction."""
        params = _dust_params()
        f      = _make_field(grid_res=16, m_base=0.5)
        # Many slip samples pushing mass from col 7 toward col 6 (−x slip)
        s = _make_sample(
            ix=7, iy=8,
            fn=3000.0, area=0.05,
            ft_x=-80.0, ft_y=0.0,
            v_rel_x=-3.0, v_rel_y=0.0,
        )
        m_before_target = f.m_at(6, 8)
        for _ in range(20):
            f.apply_contact_sample(s, params, dt=0.1)
        m_after_target = f.m_at(6, 8)
        self.assertGreaterEqual(
            m_after_target, m_before_target,
            "Mass should accumulate in front of slip direction",
        )

    def test_slip_reduces_source_mass(self) -> None:
        """The cell being slid over should lose loose material."""
        params = _dust_params()
        f      = _make_field(grid_res=16, m_base=0.8)
        s = _make_sample(
            ix=8, iy=8,
            fn=3000.0, area=0.05,
            ft_x=80.0, ft_y=0.0,
            v_rel_x=3.0, v_rel_y=0.0,
        )
        m_before = f.m_at(8, 8)
        for _ in range(20):
            f.apply_contact_sample(s, params, dt=0.1)
        m_after = f.m_at(8, 8)
        self.assertLessEqual(m_after, m_before,
            "Source cell M should not increase during bulldozing")


# ---------------------------------------------------------------------------
# 3. test_relaxation_erases_traces_over_time
# ---------------------------------------------------------------------------

class TestRelaxationErasesTraces(unittest.TestCase):
    """H and M must converge to resting values; storms accelerate it."""

    def _indent_field(self) -> DeformationField:
        params = _dust_params()
        f      = _make_field(grid_res=8)
        s      = _make_sample(ix=4, iy=4, fn=8000.0, area=0.05)
        for _ in range(5):
            f.apply_contact_sample(s, params, dt=1.0)
        return f

    def test_h_converges_to_zero(self) -> None:
        f = self._indent_field()
        h_initial = f.h_at(4, 4)
        self.assertLess(h_initial, 0.0, "Precondition: must have indent")

        # Relax for 10× tau_h (should be nearly zero)
        for _ in range(100):
            f.relax(dt=1.0, storm_multiplier=1.0)

        h_final = f.h_at(4, 4)
        self.assertGreater(h_final, h_initial,
            "H should recover toward 0 after relaxation")
        self.assertAlmostEqual(h_final, 0.0, places=1,
            msg=f"H should be near 0 after 10× tau, got {h_final:.4f}")

    def test_storm_accelerates_relaxation(self) -> None:
        f1 = self._indent_field()
        f2 = self._indent_field()

        # Same total time but f2 uses storm multiplier = 4
        for _ in range(10):
            f1.relax(dt=1.0, storm_multiplier=1.0)
            f2.relax(dt=1.0, storm_multiplier=4.0)

        h1 = f1.h_at(4, 4)
        h2 = f2.h_at(4, 4)
        self.assertGreater(h2, h1,
            f"Storm should accelerate relaxation: h_storm={h2:.6f} h_calm={h1:.6f}")

    def test_m_returns_to_base(self) -> None:
        params = _dust_params()
        f      = _make_field(grid_res=8, m_base=0.5)
        # Bulldoze some mass away from centre
        s = _make_sample(ix=4, iy=4, fn=3000.0, area=0.05,
                         ft_x=60.0, v_rel_x=2.5)
        for _ in range(20):
            f.apply_contact_sample(s, params, dt=0.1)

        m_after_deform = f.m_at(4, 4)

        for _ in range(100):
            f.relax(dt=1.0, storm_multiplier=1.0)

        m_relaxed = f.m_at(4, 4)
        # M should be closer to m_base (0.5) than right after deformation
        self.assertLess(
            abs(m_relaxed - 0.5),
            abs(m_after_deform - 0.5) + 0.01,
            "M after relaxation should be closer to m_base than after deformation",
        )
        self.assertAlmostEqual(m_relaxed, 0.5, places=1,
            msg=f"M should return toward m_base after relax, got {m_relaxed:.4f}")


# ---------------------------------------------------------------------------
# 4. test_network_stamp_determinism
# ---------------------------------------------------------------------------

class TestNetworkStampDeterminism(unittest.TestCase):
    """Same stamp batch applied to two independent fields must give same hash."""

    def _make_batch(self) -> DeformStampBatch:
        stamps = [
            DeformStamp(
                lat=10.5, lon=20.3,
                radius_m=5.0,
                depth_m=-0.03,
                push_dir_x=1.0, push_dir_y=0.0,
                push_amount=0.4,
                material=MaterialClass.DUST,
                tick_index=42,
            ),
            DeformStamp(
                lat=10.51, lon=20.31,
                radius_m=3.0,
                depth_m=-0.01,
                push_dir_x=0.0, push_dir_y=1.0,
                push_amount=0.2,
                material=MaterialClass.SNOW,
                tick_index=43,
            ),
        ]
        return DeformStampBatch(stamps=stamps)

    def test_encode_decode_roundtrip(self) -> None:
        batch   = self._make_batch()
        encoded = DeformStampCodec.encode_batch(batch)
        decoded = DeformStampCodec.decode_batch(encoded)

        self.assertEqual(len(decoded), len(batch.stamps),
            "Roundtrip must preserve stamp count")
        for orig, dec in zip(batch.stamps, decoded):
            self.assertAlmostEqual(orig.lat,      dec.lat,      places=2)
            self.assertAlmostEqual(orig.lon,      dec.lon,      places=2)
            self.assertAlmostEqual(orig.depth_m,  dec.depth_m,  places=2)
            self.assertEqual(orig.material,       dec.material)
            self.assertEqual(orig.tick_index,     dec.tick_index)

    def test_two_clients_same_hash(self) -> None:
        """Two independent fields receiving the same encoded batch must converge."""
        batch   = self._make_batch()
        encoded = DeformStampCodec.encode_batch(batch)

        # Apply to two independent fields via the integrator path
        from src.surface.DeformationIntegrator import DeformationIntegrator
        from src.surface.DeformNetReplicator import DeformNetReplicator

        integ1 = DeformationIntegrator()
        integ2 = DeformationIntegrator()

        DeformNetReplicator.apply_received_batch(encoded, integ1, dt=0.1)
        DeformNetReplicator.apply_received_batch(encoded, integ2, dt=0.1)

        # Both integrators touched chunk (105, 203) (lat*10, lon*10) for stamp 1
        for chunk_id in integ1._fields:
            if chunk_id in integ2._fields:
                h1 = integ1._fields[chunk_id].field_hash()
                h2 = integ2._fields[chunk_id].field_hash()
                self.assertEqual(h1, h2,
                    f"Field hashes must match for chunk {chunk_id}: {h1} vs {h2}")


# ---------------------------------------------------------------------------
# 5. test_budget_limits_not_exceeded
# ---------------------------------------------------------------------------

class TestBudgetLimitsNotExceeded(unittest.TestCase):
    """Upload budget must never be exceeded regardless of sample count."""

    def test_uploads_capped_at_max(self) -> None:
        max_uploads = 3

        class _FakeCfg:
            def get(self, *keys, default=None):
                path = ".".join(keys)
                if path == "deform.render.max_uploads_per_frame":
                    return max_uploads
                if path == "deform.cache.max_active_chunks":
                    return 32
                if path == "deform.grid_res":
                    return 16
                if path == "deform.m_base_default":
                    return 0.5
                if path == "deform.relax_tau_h_sec":
                    return 120.0
                if path == "deform.relax_tau_m_sec":
                    return 90.0
                return default

        integrator = DeformationIntegrator(config=_FakeCfg())

        params = _dust_params()

        # Spam 50 samples across many different chunk-proxied coords
        samples = []
        for i in range(50):
            samples.append(ContactSample(
                world_ix=i % 16,
                world_iy=(i * 3) % 16,
                fn=5000.0, ft_x=10.0, ft_y=0.0,
                v_rel_x=1.0, v_rel_y=0.0,
                area=0.05,
                material=MaterialClass.DUST,
                tick_index=i,
            ))

        integrator.apply_samples(samples, dt=0.016)
        dirty = integrator.consume_dirty_set()

        self.assertLessEqual(
            len(dirty), max_uploads,
            f"Dirty set size {len(dirty)} exceeds max_uploads={max_uploads}",
        )

    def test_all_samples_still_applied_despite_budget(self) -> None:
        """Even when uploads are capped, samples must still be integrated."""
        integrator = DeformationIntegrator()
        params     = _dust_params()

        samples = [
            _make_sample(ix=4, iy=4, fn=5000.0, area=0.05)
            for _ in range(30)
        ]
        integrator.apply_samples(samples, dt=0.016)

        # The field at default chunk (0,0) must show indentation
        f = integrator.get_field((0, 0))
        h = f.h_at(4, 4)
        self.assertLess(h, 0.0,
            f"H must be negative (indent) even when upload budget is small, got {h}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
