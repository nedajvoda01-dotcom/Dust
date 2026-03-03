"""BodyConstraintGraph — constraint-graph skeleton for one player entity.

Architecture note
-----------------
The skeleton is *not* a mesh or animation rig.  It is an abstract graph:

  Nodes  = slots / joints (with a mass and a local envelope volume)
  Edges  = constraints (distance, angle, contact anchor)
  Actuators = per-edge torque / force generators

The shell fills the envelope volumes of slots with world material.
No pre-authored skeleton exists; this graph is computed procedurally from
the CoreSeed parameters.

At this stage the graph is a data structure and replication container.
No constraint solver is wired up yet.

Public API
----------
BodyConstraintGraph(player_id, seed=0)
  .nodes          → list[BodyNode]
  .edges          → list[BodyEdge]
  .mass_kg        → float   (sum of shell slot masses)
  .body_revision  → int     (bumped on structural change)
  .update_mass(slot_masses) → None
  .to_dict()      → dict
  BodyConstraintGraph.from_dict(d) → BodyConstraintGraph
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Node / Edge data
# ---------------------------------------------------------------------------

@dataclass
class BodyNode:
    """One slot / joint in the constraint graph."""

    node_id:   int
    name:      str
    local_x:   float = 0.0  # position in local body frame
    local_y:   float = 0.0
    local_z:   float = 0.0
    mass_kg:   float = 0.0  # current shell mass (filled from world material)
    fill_frac: float = 0.0  # [0, 1] fill fraction (0 = empty, 1 = full)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id":   self.node_id,
            "name":      self.name,
            "local_x":   self.local_x,
            "local_y":   self.local_y,
            "local_z":   self.local_z,
            "mass_kg":   self.mass_kg,
            "fill_frac": self.fill_frac,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BodyNode":
        return cls(
            node_id   = int(d["node_id"]),
            name      = str(d["name"]),
            local_x   = float(d.get("local_x",   0.0)),
            local_y   = float(d.get("local_y",   0.0)),
            local_z   = float(d.get("local_z",   0.0)),
            mass_kg   = float(d.get("mass_kg",   0.0)),
            fill_frac = float(d.get("fill_frac", 0.0)),
        )


@dataclass
class BodyEdge:
    """One constraint between two nodes."""

    edge_id:    int
    node_a:     int   # node_id
    node_b:     int   # node_id
    kind:       str   # "distance" | "angle" | "contact_anchor"
    rest_value: float = 0.0  # rest length or rest angle (radians)
    stiffness:  float = 1.0  # [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id":    self.edge_id,
            "node_a":     self.node_a,
            "node_b":     self.node_b,
            "kind":       self.kind,
            "rest_value": self.rest_value,
            "stiffness":  self.stiffness,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BodyEdge":
        return cls(
            edge_id    = int(d["edge_id"]),
            node_a     = int(d["node_a"]),
            node_b     = int(d["node_b"]),
            kind       = str(d.get("kind", "distance")),
            rest_value = float(d.get("rest_value", 0.0)),
            stiffness  = float(d.get("stiffness",  1.0)),
        )


# ---------------------------------------------------------------------------
# BodyConstraintGraph
# ---------------------------------------------------------------------------

class BodyConstraintGraph:
    """Abstract constraint-graph skeleton for one player entity.

    Parameters
    ----------
    player_id :
        Server player identifier string.
    seed :
        Per-player seed (reserved for procedural skeleton variation).
    """

    def __init__(self, player_id: str = "", seed: int = 0) -> None:
        self._player_id     = str(player_id)
        self._seed          = int(seed)
        self._body_revision = 0

        self.nodes: List[BodyNode] = []
        self.edges: List[BodyEdge] = []

        self._build_default_skeleton()

    # ------------------------------------------------------------------
    def _build_default_skeleton(self) -> None:
        """Create a minimal bipedal constraint graph (data only, no solver)."""
        slots = [
            (0, "core",         0.00,  0.00, 0.00),
            (1, "torso",        0.00,  0.50, 0.00),
            (2, "head",         0.00,  1.00, 0.00),
            (3, "arm_l",       -0.30,  0.60, 0.00),
            (4, "arm_r",        0.30,  0.60, 0.00),
            (5, "leg_l",       -0.15, -0.50, 0.00),
            (6, "leg_r",        0.15, -0.50, 0.00),
        ]
        for nid, name, lx, ly, lz in slots:
            self.nodes.append(BodyNode(node_id=nid, name=name,
                                       local_x=lx, local_y=ly, local_z=lz))

        connections = [
            (0, 0, 1, "distance", 0.50),
            (1, 1, 2, "distance", 0.50),
            (2, 1, 3, "distance", 0.35),
            (3, 1, 4, "distance", 0.35),
            (4, 0, 5, "distance", 0.50),
            (5, 0, 6, "distance", 0.50),
        ]
        for eid, na, nb, kind, rest in connections:
            self.edges.append(BodyEdge(edge_id=eid, node_a=na, node_b=nb,
                                       kind=kind, rest_value=rest))

    # ------------------------------------------------------------------
    @property
    def body_revision(self) -> int:
        return self._body_revision

    @property
    def mass_kg(self) -> float:
        """Total shell mass (sum of per-node mass)."""
        return sum(n.mass_kg for n in self.nodes)

    # ------------------------------------------------------------------
    def update_mass(self, slot_masses: Dict[str, float]) -> None:
        """Update per-node masses and bump body_revision.

        Parameters
        ----------
        slot_masses :
            ``{node_name: mass_kg}`` mapping.  Missing names keep current mass.
        """
        changed = False
        for node in self.nodes:
            if node.name in slot_masses:
                new_mass = float(slot_masses[node.name])
                if abs(node.mass_kg - new_mass) > 1e-6:
                    node.mass_kg   = new_mass
                    node.fill_frac = min(1.0, new_mass / max(0.01, 5.0))
                    changed = True
        if changed:
            self._body_revision += 1

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id":     self._player_id,
            "seed":          self._seed,
            "body_revision": self._body_revision,
            "nodes":         [n.to_dict() for n in self.nodes],
            "edges":         [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BodyConstraintGraph":
        g = cls(player_id=str(d.get("player_id", "")),
                seed=int(d.get("seed", 0)))
        g._body_revision = int(d.get("body_revision", 0))
        g.nodes = [BodyNode.from_dict(n) for n in d.get("nodes", [])]
        g.edges = [BodyEdge.from_dict(e) for e in d.get("edges", [])]
        return g
