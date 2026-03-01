"""test_planet_time — Stage 7 PlanetTimeSystem and InsolationField tests.

Tests
-----
1. TestLocalSolarTime              — longitude offset of 180° → solar time
                                     offset ≈ 0.5; period = dayLengthSeconds
2. TestInsolationFieldDeterminism  — same seed + time → identical field hash
3. TestRingShadowBandMotion        — ring shadow at fixed lat/lon varies over
                                     planet rotation (shadow band crawls)
4. TestEclipseSignature            — eclipse → reduced directTotal vs. no-eclipse
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.Config import Config
from src.math.PlanetMath import PlanetMath, LatLong
from src.math.Vec3 import Vec3
from src.systems.AstroSystem import AstroSystem
from src.systems.InsolationField import InsolationField, InsolCell
from src.systems.PlanetTimeSystem import PlanetTimeSystem

_PLANET_R = 1000.0
_W, _H = 16, 8   # small grid for fast test execution


def _make_systems() -> tuple[Config, AstroSystem, PlanetTimeSystem]:
    config = Config()
    astro = AstroSystem(config)
    pts = PlanetTimeSystem(config, astro)
    return config, astro, pts


def _make_field(config: Config) -> InsolationField:
    return InsolationField(config, width=_W, height=_H)


# ---------------------------------------------------------------------------
# TestLocalSolarTime
# ---------------------------------------------------------------------------

class TestLocalSolarTime(unittest.TestCase):
    """Local solar time correctness: geometry and periodicity."""

    def test_longitude_offset_180_gives_half_day(self) -> None:
        """Two equatorial points 180° apart must have solar-time offset ≈ 0.5."""
        config, astro, pts = _make_systems()
        astro.update(0.0)

        # pos1 → lon = atan2(0, 1) = 0
        pos1 = Vec3(0.0, 0.0, 1.0)
        # pos2 → lon = atan2(0, -1) = π
        pos2 = Vec3(0.0, 0.0, -1.0)

        t1 = pts.get_local_solar_time_01(pos1, 0.0)
        t2 = pts.get_local_solar_time_01(pos2, 0.0)

        diff = abs(t2 - t1)
        diff = min(diff, 1.0 - diff)   # wrap-around distance on [0,1)
        self.assertAlmostEqual(
            diff, 0.5, delta=0.01,
            msg=f"180° longitude offset should give Δsolar_time≈0.5; got {diff:.4f}",
        )

    def test_solar_time_returns_after_full_day(self) -> None:
        """After exactly one day length, local solar time must equal its t=0 value."""
        config, astro, pts = _make_systems()
        pos = Vec3(1.0, 0.0, 0.0)   # equatorial, lon = π/2

        t0 = pts.get_local_solar_time_01(pos, 0.0)
        t1 = pts.get_local_solar_time_01(pos, pts._day_len_s)

        diff = abs(t1 - t0)
        diff = min(diff, 1.0 - diff)
        self.assertAlmostEqual(
            diff, 0.0, delta=1e-6,
            msg=f"Solar time after one full day should equal t=0 value; Δ={diff:.2e}",
        )

    def test_is_day_and_is_night_are_exclusive(self) -> None:
        """is_day and is_night must be mutually exclusive for any surface normal."""
        config, astro, pts = _make_systems()
        astro.update(0.0)

        normals = [
            Vec3(0.0, 0.0, 1.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(-1.0, 0.0, 0.0),
            Vec3(1.0, 0.0, 0.0),
        ]
        for n in normals:
            day = pts.is_day(n)
            night = pts.is_night(n)
            self.assertNotEqual(
                day, night,
                msg=f"is_day={day} and is_night={night} must differ for normal {n}",
            )


# ---------------------------------------------------------------------------
# TestInsolationFieldDeterminism
# ---------------------------------------------------------------------------

class TestInsolationFieldDeterminism(unittest.TestCase):
    """Identical config + time must produce identical field data."""

    @staticmethod
    def _make_computed_field(t: float) -> InsolationField:
        config = Config()
        planet_r = config.get("planet", "radius_units", default=_PLANET_R)
        astro = AstroSystem(config)
        astro.update(t)
        field = _make_field(config)
        field.force_full_update(t, astro, planet_r)
        return field

    @staticmethod
    def _field_hash(field: InsolationField) -> str:
        vals = [round(v, 8) for v in field._prev.direct_total]
        return hashlib.md5(str(vals).encode()).hexdigest()

    def test_same_inputs_produce_same_field(self) -> None:
        """Two independently created fields at the same time must be identical."""
        t = 1234.5
        h1 = self._field_hash(self._make_computed_field(t))
        h2 = self._field_hash(self._make_computed_field(t))
        self.assertEqual(h1, h2, "InsolationField must be deterministic for same config/time")

    def test_different_times_produce_different_fields(self) -> None:
        """Fields at different times should differ (basic sanity check)."""
        h1 = self._field_hash(self._make_computed_field(0.0))
        # Advance by one quarter day so day/night shifts
        day_s = Config().get("day", "length_minutes", default=90) * 60.0
        h2 = self._field_hash(self._make_computed_field(day_s * 0.25))
        self.assertNotEqual(h1, h2, "Fields at different times should differ")


# ---------------------------------------------------------------------------
# TestRingShadowBandMotion
# ---------------------------------------------------------------------------

class TestRingShadowBandMotion(unittest.TestCase):
    """Ring shadow at a fixed surface location must vary as the planet rotates."""

    def test_ring_shadow_varies_over_full_day(self) -> None:
        """A cell near the ring inner-edge transition latitude goes from shadow≈0
        (spin=0, r < ring_inner) to shadow≈1 (spin=π/2, r well inside ring band).

        Grid 64×32 gives row y=12 at lat ≈ -19.7°, which is analytically just
        below ring_inner at spin=0 (r≈1392) and well inside the band at spin=π/2
        (r≈1681), producing a range > 0.5 over one full rotation.
        """
        config = Config()
        planet_r = config.get("planet", "radius_units", default=_PLANET_R)
        day_s = config.get("day", "length_minutes", default=90) * 60.0
        astro = AstroSystem(config)

        # 64×32 grid — default config resolution; needed to resolve the transition cell
        field = InsolationField(config, width=64, height=32)

        # Exact centre of row y=12 in a 64×32 grid: lat ≈ -19.7°
        lat_transition = -math.pi / 2.0 + math.pi * 12.5 / 32.0
        planet_frame_pos = PlanetMath.direction_from_lat_long(
            LatLong(lat_transition, 0.0)
        )

        shadow_vals: list[float] = []
        steps = 36
        for i in range(steps):
            t = day_s * i / steps
            astro.update(t)
            field.force_full_update(t, astro, planet_r)
            cell = field.sample_at(planet_frame_pos)
            shadow_vals.append(cell.ring_shadow_eff)

        max_s = max(shadow_vals)
        min_s = min(shadow_vals)
        self.assertGreater(
            max_s - min_s, 0.5,
            msg=(
                f"Ring shadow should vary significantly over a full day; "
                f"max={max_s:.3f}, min={min_s:.3f}, range={max_s - min_s:.3f}"
            ),
        )

    def test_ring_shadow_periodicity(self) -> None:
        """Ring shadow pattern repeats after one full planet rotation."""
        config = Config()
        planet_r = config.get("planet", "radius_units", default=_PLANET_R)
        day_s = config.get("day", "length_minutes", default=90) * 60.0
        astro = AstroSystem(config)
        field = InsolationField(config, width=64, height=32)

        lat_transition = -math.pi / 2.0 + math.pi * 12.5 / 32.0
        planet_frame_pos = PlanetMath.direction_from_lat_long(
            LatLong(lat_transition, 0.0)
        )

        # Sample at t=0 and t=dayLength — should be the same spin angle
        def _shadow_at(t: float) -> float:
            astro.update(t)
            field.force_full_update(t, astro, planet_r)
            return field.sample_at(planet_frame_pos).ring_shadow_eff

        s_start = _shadow_at(0.0)
        s_end = _shadow_at(day_s)
        self.assertAlmostEqual(
            s_start, s_end, delta=1e-4,
            msg=f"Ring shadow should repeat after one full day; start={s_start:.5f}, end={s_end:.5f}",
        )


# ---------------------------------------------------------------------------
# TestEclipseSignature
# ---------------------------------------------------------------------------

class TestEclipseSignature(unittest.TestCase):
    """Eclipse event must leave a measurable depression in total insolation."""

    def _avg_total(self, field: InsolationField) -> float:
        """Average directTotal across all cells (including night side)."""
        total = sum(field._prev.direct_total)
        return total / field._n

    def test_eclipse_factor_is_elevated(self) -> None:
        """At t=0 (suns aligned along +Z) the eclipse factor must be > 0."""
        astro = AstroSystem(Config())
        astro.update(0.0)
        self.assertGreater(
            astro.get_eclipse_factor(), 0.0,
            msg="Eclipse factor should be > 0 when suns are aligned",
        )

    def test_eclipse_reduces_average_insolation(self) -> None:
        """Average directTotal during eclipse must be lower than outside eclipse."""
        config = Config()
        planet_r = config.get("planet", "radius_units", default=_PLANET_R)
        bp_s = config.get("binary", "period_minutes", default=18) * 60.0

        # --- eclipse moment: t=0, both suns along +Z, max overlap ---
        astro = AstroSystem(config)
        astro.update(0.0)
        field_eclipse = _make_field(config)
        field_eclipse.force_full_update(0.0, astro, planet_r)

        # --- no-eclipse moment: t = binary_period/4, suns maximally separated ---
        astro.update(bp_s * 0.25)
        field_clear = _make_field(config)
        field_clear.force_full_update(bp_s * 0.25, astro, planet_r)

        avg_eclipse = self._avg_total(field_eclipse)
        avg_clear = self._avg_total(field_clear)

        self.assertGreater(avg_clear, 0.0,
                           msg="Away from eclipse, average directTotal should be positive")
        self.assertLess(
            avg_eclipse, avg_clear,
            msg=(
                f"Eclipse should reduce avg directTotal: "
                f"eclipse={avg_eclipse:.4f}, clear={avg_clear:.4f}"
            ),
        )


# ---------------------------------------------------------------------------
# TestTiledUpdate
# ---------------------------------------------------------------------------

class TestTiledUpdate(unittest.TestCase):
    """Tiled update must produce the same result as force_full_update."""

    def test_tiled_equals_full_update(self) -> None:
        """After enough ticks the tiled buffer should match force_full_update."""
        config = Config()
        planet_r = config.get("planet", "radius_units", default=_PLANET_R)
        astro = AstroSystem(config)
        astro.update(0.0)

        # Tiled field: cells_per_tick=1 to exercise the cursor logic
        field_tiled = InsolationField(config, width=_W, height=_H)
        field_tiled._cells_per_tick = 1

        # Drive enough ticks to complete one full pass (W*H ticks)
        # first tick triggers swap of the empty initial buffer → _prev becomes
        # the initial zeros; subsequent ticks fill _next.
        # We need W*H more ticks to fill _next completely.
        total_cells = _W * _H
        for _ in range(total_cells + 2):
            field_tiled.update(0.0, astro, planet_r)

        # Reference: force_full_update
        field_ref = InsolationField(config, width=_W, height=_H)
        field_ref.force_full_update(0.0, astro, planet_r)

        # Compare _prev buffers (rounded to avoid float ordering noise)
        for i in range(total_cells):
            self.assertAlmostEqual(
                field_tiled._prev.direct_total[i],
                field_ref._prev.direct_total[i],
                delta=1e-9,
                msg=f"Cell {i}: tiled={field_tiled._prev.direct_total[i]:.6f} "
                    f"ref={field_ref._prev.direct_total[i]:.6f}",
            )


if __name__ == "__main__":
    unittest.main()
