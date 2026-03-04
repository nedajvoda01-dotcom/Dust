"""MaterialGraph — Stage 63 canonical Planet Reality material node graph.

Defines the fixed topology of the planetary material system (v1.0).

Constants
---------
MATERIAL_GRAPH_VERSION : int
    Version lock.  Any topology change requires a world epoch increment.

MaterialNode : IntEnum
    All valid material states (nodes) of the planet and character.

PHASE_TRANSITIONS : frozenset
    Fixed set of valid (from_node, to_node) phase-transition edges.

TickOrder : namespace
    Fixed simulation update order (integers 1–7).

Notes
-----
* The AI must not modify graph topology without a migration.
* Adding a new MaterialNode or edge requires incrementing
  MATERIAL_GRAPH_VERSION and worldEpoch.
"""
from __future__ import annotations

import enum


# ---------------------------------------------------------------------------
# Version lock
# ---------------------------------------------------------------------------

MATERIAL_GRAPH_VERSION: int = 1
"""Topology version.  Increment on any node/edge addition or removal."""


# ---------------------------------------------------------------------------
# Material nodes (graph vertices)
# ---------------------------------------------------------------------------

class MaterialNode(enum.IntEnum):
    """All material states recognised by the Planet Reality graph (v1.0).

    Solid
    -----
    SOLID_ROCK        — deep bedrock
    CRUST             — surface crust layer
    DEBRIS_FRAGMENTS  — coarse breakage material

    Granular
    --------
    REGOLITH_DUST     — fine surface granular layer
    SNOW_LOOSE        — uncompacted snow
    SNOW_COMPACTED    — pressure-consolidated snow

    Film
    ----
    ICE_FILM          — thin ice coating

    Liquid
    ------
    MAGMA             — active melt (vent-local only)
    WATER_RARE        — rare liquid water (low default contribution)

    Gas / aerosol
    -------------
    VAPOR             — evaporated water / sublimated ice
    AEROSOL_DUST      — airborne fine dust

    Character special
    -----------------
    STRUCTURAL_CORE   — character body core; exempt from erosion
    """
    SOLID_ROCK       = 0
    CRUST            = 1
    DEBRIS_FRAGMENTS = 2
    REGOLITH_DUST    = 3
    SNOW_LOOSE       = 4
    SNOW_COMPACTED   = 5
    ICE_FILM         = 6
    MAGMA            = 7
    WATER_RARE       = 8
    VAPOR            = 9
    AEROSOL_DUST     = 10
    STRUCTURAL_CORE  = 11


# ---------------------------------------------------------------------------
# Phase transitions (graph edges) — fixed for v1.0
# ---------------------------------------------------------------------------

PHASE_TRANSITIONS: frozenset = frozenset([
    # 4.1 Deposition / Erosion
    (MaterialNode.AEROSOL_DUST,      MaterialNode.REGOLITH_DUST),
    (MaterialNode.SNOW_LOOSE,        MaterialNode.AEROSOL_DUST),
    (MaterialNode.AEROSOL_DUST,      MaterialNode.SNOW_LOOSE),
    (MaterialNode.CRUST,             MaterialNode.DEBRIS_FRAGMENTS),
    (MaterialNode.DEBRIS_FRAGMENTS,  MaterialNode.REGOLITH_DUST),

    # 4.2 Compaction
    (MaterialNode.SNOW_LOOSE,        MaterialNode.SNOW_COMPACTED),
    (MaterialNode.REGOLITH_DUST,     MaterialNode.CRUST),

    # 4.3 Melt / Freeze
    (MaterialNode.SNOW_LOOSE,        MaterialNode.WATER_RARE),
    (MaterialNode.ICE_FILM,          MaterialNode.WATER_RARE),
    (MaterialNode.WATER_RARE,        MaterialNode.ICE_FILM),
    (MaterialNode.WATER_RARE,        MaterialNode.VAPOR),
    (MaterialNode.VAPOR,             MaterialNode.ICE_FILM),
    (MaterialNode.VAPOR,             MaterialNode.SNOW_LOOSE),

    # 4.4 Fracture / Instability (slope flow)
    (MaterialNode.DEBRIS_FRAGMENTS,  MaterialNode.REGOLITH_DUST),

    # 4.5 Magma transitions
    (MaterialNode.MAGMA,             MaterialNode.CRUST),
    (MaterialNode.CRUST,             MaterialNode.MAGMA),
])
"""Allowed (source, target) material-state transitions.  Read-only."""


def is_valid_transition(
    from_node: MaterialNode,
    to_node: MaterialNode,
) -> bool:
    """Return True if *from_node* → *to_node* is a legal phase transition."""
    return (from_node, to_node) in PHASE_TRANSITIONS


# ---------------------------------------------------------------------------
# Tick order — fixed simulation update sequence
# ---------------------------------------------------------------------------

class TickOrder:
    """Fixed per-tick update priorities (lower = earlier).

    Attributes
    ----------
    ATMOSPHERE : int = 1
    MICROCLIMATE : int = 2
    PHASE_TRANSITIONS : int = 3
    CHARACTER_TO_WORLD : int = 4
    INSTABILITY : int = 5
    MEMORY_COMPACTION : int = 6
    ENERGY_NORMALISATION : int = 7
    """
    ATMOSPHERE         = 1
    MICROCLIMATE       = 2
    PHASE_TRANSITIONS  = 3
    CHARACTER_TO_WORLD = 4
    INSTABILITY        = 5
    MEMORY_COMPACTION  = 6
    ENERGY_NORMALISATION = 7
