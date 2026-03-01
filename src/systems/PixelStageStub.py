"""IPixelStage stub — interface for future pixel-art post-process pass."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class IPixelStage(ABC):
    @abstractmethod
    def apply(self, render_target: Any) -> None:
        """Apply pixel-art post-processing to render_target. Stub."""
        ...


class PixelStageStub(IPixelStage):
    def apply(self, render_target: Any) -> None:
        pass
