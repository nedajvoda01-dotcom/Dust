"""Entity — basic scene node with a Transform and optional components."""
from __future__ import annotations

from typing import Any

from src.scene.Transform import Transform


class Entity:
    _next_id: int = 0

    def __init__(self, name: str = "Entity") -> None:
        self.id: int = Entity._next_id
        Entity._next_id += 1
        self.name: str = name
        self.transform: Transform = Transform()
        self._components: dict[type, Any] = {}
        self.active: bool = True

    def add_component(self, component: Any) -> None:
        self._components[type(component)] = component

    def get_component(self, cls: type) -> Any | None:
        return self._components.get(cls)

    def has_component(self, cls: type) -> bool:
        return cls in self._components

    def __repr__(self) -> str:
        return f"Entity(id={self.id}, name={self.name!r})"
