"""SkyPrimitivesRenderer — Stage 6 minimal sky-object renderer.

Draws procedurally (no textures):
- Two sun discs (filled triangle fan)
- Ring annulus (wireframe strip)
- Moon disc (filled triangle fan)

All objects are rendered as 3-D billboards placed on a large sky sphere so
they appear at infinity.  No atmospheric scattering or post-effects.
"""
from __future__ import annotations

import math

from src.math.Vec3 import Vec3
from src.systems.AstroSystem import AstroSystem

_SUN1_COLOR = (1.0, 0.90, 0.60)   # warm yellow-orange
_SUN2_COLOR = (0.70, 0.85, 1.00)  # cool blue-white
_MOON_COLOR = (0.75, 0.72, 0.65)  # pale grey
_RING_COLOR  = (0.62, 0.55, 0.42, 0.45)  # dusty tan, semi-transparent

_TWO_PI = 2.0 * math.pi
_SKY_RADIUS = 9000.0   # units — large enough to appear "at infinity"
_SUN_DISC_HALF_ANGLE = math.radians(3.5)  # visual disc half-angle (exaggerated for visibility)
_MOON_DISC_HALF_ANGLE = math.radians(2.0)

_DISC_SEGMENTS = 24
_RING_SEGMENTS = 64


class SkyPrimitivesRenderer:
    """Renders sky objects using the AstroSystem state.

    Call :meth:`render` once per frame after all 3-D geometry has been drawn.
    Requires an active OpenGL context (provided by the main Renderer).
    """

    def __init__(self, astro: AstroSystem) -> None:
        self._astro = astro

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def render(self, camera_pos: Vec3) -> None:
        """Draw all sky primitives relative to camera_pos."""
        try:
            from OpenGL.GL import (  # noqa: PLC0415
                glBegin, glEnd, glVertex3f, glColor3f, glColor4f,
                glDisable, glEnable, glBlendFunc, glDepthMask,
                GL_TRIANGLES, GL_TRIANGLE_FAN, GL_LINES,
                GL_LIGHTING, GL_DEPTH_TEST, GL_BLEND,
                GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
            )
        except ImportError:
            return  # no OpenGL available (e.g. tests)

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(False)

        dir1, dir2 = self._astro.get_sun_directions()
        moon_dir = (self._astro.moon_world_pos).normalized()

        self._draw_disc(camera_pos, dir1, _SUN_DISC_HALF_ANGLE, _SUN1_COLOR)
        self._draw_disc(camera_pos, dir2, _SUN_DISC_HALF_ANGLE, _SUN2_COLOR)
        self._draw_disc(camera_pos, moon_dir, _MOON_DISC_HALF_ANGLE, _MOON_COLOR)
        self._draw_ring(camera_pos)

        glDepthMask(True)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _disc_basis(center_dir: Vec3) -> tuple[Vec3, Vec3]:
        """Return two orthogonal vectors perpendicular to center_dir."""
        d = center_dir.normalized()
        if abs(d.y) < 0.9:
            ref = Vec3(0.0, 1.0, 0.0)
        else:
            ref = Vec3(1.0, 0.0, 0.0)
        right = ref.cross(d).normalized()
        up = d.cross(right).normalized()
        return right, up

    def _draw_disc(
        self,
        origin: Vec3,
        direction: Vec3,
        half_angle: float,
        color: tuple,
    ) -> None:
        """Draw a filled disc as a triangle fan at sky radius."""
        try:
            from OpenGL.GL import glBegin, glEnd, glVertex3f, glColor3f, GL_TRIANGLE_FAN  # noqa: PLC0415
        except ImportError:
            return

        centre = origin + direction.normalized() * _SKY_RADIUS
        r = _SKY_RADIUS * math.tan(half_angle)
        right, up = self._disc_basis(direction)

        glColor3f(*color)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(centre.x, centre.y, centre.z)
        for i in range(_DISC_SEGMENTS + 1):
            a = _TWO_PI * i / _DISC_SEGMENTS
            p = centre + right * (math.cos(a) * r) + up * (math.sin(a) * r)
            glVertex3f(p.x, p.y, p.z)
        glEnd()

    def _draw_ring(self, origin: Vec3) -> None:
        """Draw the ring as a flat annulus strip around the planet."""
        try:
            from OpenGL.GL import glBegin, glEnd, glVertex3f, glColor4f, GL_TRIANGLES  # noqa: PLC0415
        except ImportError:
            return

        astro = self._astro
        # Ring geometry in world space — centre at planet origin
        re1 = astro._re1  # noqa: SLF001
        re2 = astro._re2  # noqa: SLF001
        inner = astro._ring_inner
        outer = astro._ring_outer

        r, g, b, a = _RING_COLOR
        glColor4f(r, g, b, a)
        glBegin(GL_TRIANGLES)
        for i in range(_RING_SEGMENTS):
            a0 = _TWO_PI * i / _RING_SEGMENTS
            a1 = _TWO_PI * (i + 1) / _RING_SEGMENTS
            pi = re1 * (math.cos(a0) * inner) + re2 * (math.sin(a0) * inner)
            po = re1 * (math.cos(a0) * outer) + re2 * (math.sin(a0) * outer)
            pi2 = re1 * (math.cos(a1) * inner) + re2 * (math.sin(a1) * inner)
            po2 = re1 * (math.cos(a1) * outer) + re2 * (math.sin(a1) * outer)
            # two triangles per strip segment
            for v in (pi, po, pi2):
                glVertex3f(v.x, v.y, v.z)
            for v in (po, po2, pi2):
                glVertex3f(v.x, v.y, v.z)
        glEnd()

