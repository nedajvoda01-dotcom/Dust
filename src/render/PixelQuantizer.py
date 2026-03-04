"""PixelQuantizer — Stage 62 pixel-grid quantization pass.

Implements the Pixel Quantization Pass described in §3.4:

* Downscale the full-resolution linear buffer to a *virtual pixel resolution*
  (e.g. 480 p), then upscale back with a nearest/box filter so individual
  pixels are visually enlarged.
* The pixel grid is anchored to **screen space** (not world space) so it
  never drifts under camera movement.
* Subpixel jitter is suppressed: each sample is snapped to the nearest
  virtual-pixel centre before any filtering.

All computation is pure Python / NumPy-free math — the module operates on
flat ``list[tuple[float,float,float]]`` colour buffers so it can be unit-
tested without a GPU context.

Public API
----------
PixelQuantizerConfig (dataclass)
PixelQuantizer
  .quantize(buffer, src_w, src_h) → list[tuple[float,float,float]]
  .pixel_width  : int   — virtual pixel width
  .pixel_height : int   — virtual pixel height
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

_Pixel = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PixelQuantizerConfig:
    """Parameters for the quantization pass (§12 config keys).

    Attributes
    ----------
    internal_resolution : tuple[int, int]
        Full 3-D render resolution (width, height) — e.g. (1920, 1080).
    pixel_resolution : tuple[int, int]
        Virtual low-res grid (width, height) — e.g. (853, 480).
    pixel_scale_mode : str
        ``"nearest"``  — hard pixel grid (default, recommended).
        ``"box"``       — 2×2 box average before snap (softer).
    """
    internal_resolution: Tuple[int, int] = (1920, 1080)
    pixel_resolution: Tuple[int, int] = (853, 480)
    pixel_scale_mode: str = "nearest"


# ---------------------------------------------------------------------------
# PixelQuantizer
# ---------------------------------------------------------------------------

class PixelQuantizer:
    """Downscale → snap → upscale pixel quantization.

    Parameters
    ----------
    config :
        :class:`PixelQuantizerConfig` instance.  Passing ``None`` uses
        the dataclass defaults.
    """

    def __init__(self, config: PixelQuantizerConfig | None = None) -> None:
        self._cfg = config or PixelQuantizerConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pixel_width(self) -> int:
        return self._cfg.pixel_resolution[0]

    @property
    def pixel_height(self) -> int:
        return self._cfg.pixel_resolution[1]

    @property
    def internal_width(self) -> int:
        return self._cfg.internal_resolution[0]

    @property
    def internal_height(self) -> int:
        return self._cfg.internal_resolution[1]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def quantize(
        self,
        buffer: List[_Pixel],
        src_w: int,
        src_h: int,
    ) -> List[_Pixel]:
        """Apply the pixel-quantization pass to *buffer*.

        Parameters
        ----------
        buffer :
            Flat list of (r, g, b) tuples in row-major order, length
            ``src_w * src_h``.
        src_w, src_h :
            Dimensions of *buffer*.

        Returns
        -------
        list[tuple[float, float, float]]
            Upscaled buffer with dimensions ``src_w × src_h`` where each
            virtual pixel is a uniform colour block.
        """
        pw = self.pixel_width
        ph = self.pixel_height

        # Step 1: downscale src_w×src_h → pw×ph (nearest or box)
        low_res = self._downscale(buffer, src_w, src_h, pw, ph)

        # Step 2: upscale pw×ph → src_w×src_h (nearest — stable grid)
        out = self._upscale_nearest(low_res, pw, ph, src_w, src_h)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _downscale(
        self,
        buf: List[_Pixel],
        sw: int, sh: int,
        dw: int, dh: int,
    ) -> List[_Pixel]:
        """Downscale *buf* (sw×sh) → (dw×dh).

        Uses nearest sampling to avoid blurring the pixel grid.
        For ``pixel_scale_mode == "box"`` a 2×2 average is used instead.
        """
        out: List[_Pixel] = []
        mode = self._cfg.pixel_scale_mode

        for dy in range(dh):
            for dx in range(dw):
                # Map destination pixel centre back to source space
                sx_f = (dx + 0.5) * sw / dw
                sy_f = (dy + 0.5) * sh / dh

                if mode == "box":
                    # 2×2 box average around nearest source pixel
                    r, g, b = self._box2(buf, sw, sh, sx_f, sy_f)
                else:
                    # Nearest: snap to pixel centre
                    sx = int(sx_f)
                    sy = int(sy_f)
                    sx = max(0, min(sw - 1, sx))
                    sy = max(0, min(sh - 1, sy))
                    r, g, b = buf[sy * sw + sx]
                out.append((r, g, b))
        return out

    def _box2(
        self,
        buf: List[_Pixel],
        sw: int, sh: int,
        cx: float, cy: float,
    ) -> _Pixel:
        """2×2 box average centred at (cx, cy) in source space."""
        x0 = max(0, min(sw - 1, int(cx - 0.5)))
        y0 = max(0, min(sh - 1, int(cy - 0.5)))
        x1 = max(0, min(sw - 1, x0 + 1))
        y1 = max(0, min(sh - 1, y0 + 1))

        c00 = buf[y0 * sw + x0]
        c10 = buf[y0 * sw + x1]
        c01 = buf[y1 * sw + x0]
        c11 = buf[y1 * sw + x1]
        return (
            (c00[0] + c10[0] + c01[0] + c11[0]) * 0.25,
            (c00[1] + c10[1] + c01[1] + c11[1]) * 0.25,
            (c00[2] + c10[2] + c01[2] + c11[2]) * 0.25,
        )

    @staticmethod
    def _upscale_nearest(
        low: List[_Pixel],
        lw: int, lh: int,
        dw: int, dh: int,
    ) -> List[_Pixel]:
        """Upscale *low* (lw×lh) → (dw×dh) with nearest (stable grid)."""
        out: List[_Pixel] = []
        for dy in range(dh):
            for dx in range(dw):
                # Snap to virtual pixel centre — no sub-pixel drift
                lx = int(dx * lw / dw)
                ly = int(dy * lh / dh)
                lx = max(0, min(lw - 1, lx))
                ly = max(0, min(lh - 1, ly))
                out.append(low[ly * lw + lx])
        return out
