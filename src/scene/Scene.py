"""Scene — flat entity registry and update loop."""
from __future__ import annotations

from typing import Iterator

from src.scene.Entity import Entity


class Scene:
    def __init__(self, name: str = "Scene") -> None:
        self.name: str = name
        self._entities: dict[int, Entity] = {}

    def add(self, entity: Entity) -> Entity:
        self._entities[entity.id] = entity
        return entity

    def remove(self, entity: Entity) -> None:
        self._entities.pop(entity.id, None)

    def get(self, entity_id: int) -> Entity | None:
        return self._entities.get(entity_id)

    def all(self) -> Iterator[Entity]:
        yield from self._entities.values()

    def active_entities(self) -> Iterator[Entity]:
        for e in self._entities.values():
            if e.active:
                yield e

    def __len__(self) -> int:
        return len(self._entities)
