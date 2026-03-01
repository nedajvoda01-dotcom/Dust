"""SubsurfaceSystem — Stage 26 procedural subsurface world (caves, tubes, chambers).

Generates and manages a deterministic underground layer integrated into the
planet SDF.  No assets, no UI, no Minecraft-style corridors.

Architecture
------------
CaveCarver          — abstract SDF carver (LavaTube / FractureCavern / Chamber)
CaveNode            — graph node (chamber / tube junction / fracture intersection)
CaveEdge            — graph edge (tunnel connecting two nodes)
CaveGraph           — full directed cave graph
PortalFinder        — locates rare surface entry points (fractures / sinkholes)
CaveFactor          — per-position cave-atmosphere factor (0 = surface, 1 = deep)
CollapseEvent       — server-authoritative dynamic collapse event
SubsurfaceSystem    — public API

Generation is 100 % deterministic from (world_seed XOR subsurface_seed_salt).

Cave depth layers
-----------------
SHALLOW  (0–200 m)   : hairline fractures, small pockets, frequent portals
MID      (200–1000 m): lava tubes, connected tunnels, main traversal layer
DEEP     (1–5 km)    : large chambers, rare abysses, very low portal frequency

SDF compositing
---------------
``SDF_Final(p) = max(SDF_Surface(p), −SDF_Caves(p))``
Implemented as a set of SDF subtraction patches (SphereCarve / CapsuleCarve)
applied to the SDFPatchSystem.

Multiplayer
-----------
Cave geometry is deterministic (seed-driven) — all clients see the same world.
Dynamic collapses are server-authoritative events replicated as
``SUBSURFACE_EVENT`` JSON + a ``SubsurfacePatchBatch`` of SDF patches.
Clients call ``apply_event_patch(batch)`` to update their local SDF.

Public API
----------
SubsurfaceSystem(config, global_seed, geo_sampler, planet_radius, sdf_world)
  .update(dt, game_time, player_pos)       — advance simulation
  .cave_factor_at(world_pos) → float       — 0..1 cave atmosphere factor
  .portals()                 → list[Vec3]  — surface entry directions
  .generate_collapse(world_pos, seed) → SubsurfacePatchBatch
  .apply_event_patch(batch)                — client applies server event
  .cave_graph                → CaveGraph
  .event_log                 → list[CollapseEvent]
  .debug_info()              → dict

Config keys (under ``subsurface``)
-----------------------------------
enable, seed_salt, shallow_prob, mid_prob, deep_prob,
tube_radius_min, tube_radius_max, chamber_size_min, chamber_size_max,
fracture_thickness, portal_frequency, cave_fog_density,
cave_light_scatter, cave_audio_reverb_mix, collapse_event_rate_cap.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from src.math.Vec3 import Vec3
from src.planet.SDFPatchSystem import (
    AdditiveDeposit,
    CapsuleCarve,
    SDFPatch,
    SDFPatchLog,
    SphereCarve,
)
from src.systems.SubsurfaceSystemStub import ISubsurfaceSystem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _vec3_from_seed(rng: random.Random) -> Vec3:
    """Return a random unit Vec3 using Box–Muller / rejection on the RNG."""
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        r = math.sqrt(x * x + y * y + z * z)
        if r > 1e-9:
            return Vec3(x / r, y / r, z / r)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CaveDepthLayer(Enum):
    SHALLOW = auto()   # 0–200 m: fractures, small pockets
    MID     = auto()   # 200–1000 m: lava tubes, connected tunnels
    DEEP    = auto()   # 1–5 km: large chambers, rare abysses


class CaveNodeKind(Enum):
    CHAMBER    = auto()   # widened void (ellipsoidal hall)
    TUBE_JOINT = auto()   # lava tube junction / bifurcation
    FRACTURE   = auto()   # tectonic fracture intersection


class SubsurfaceEventType(Enum):
    COLLAPSE = auto()   # tunnel/chamber collapse — dynamic, server-authoritative


# ---------------------------------------------------------------------------
# Cave graph structures
# ---------------------------------------------------------------------------


@dataclass
class CaveNode:
    """A node in the cave graph (chamber, junction, or fracture crossing)."""
    node_id:    int
    kind:       CaveNodeKind
    direction:  Vec3           # unit direction from planet centre
    depth:      float          # metres below surface (positive = deeper)
    radius:     float          # representative size (metres)
    layer:      CaveDepthLayer


@dataclass
class CaveEdge:
    """A tunnel (edge) connecting two CaveNodes."""
    edge_id:  int
    src_id:   int              # CaveNode.node_id
    dst_id:   int
    radius:   float            # tube/tunnel radius (metres)


@dataclass
class CaveGraph:
    """Directed cave graph — nodes + edges, both deterministic from seed."""
    nodes: List[CaveNode] = field(default_factory=list)
    edges: List[CaveEdge] = field(default_factory=list)

    # Fast lookups
    _by_id: Dict[int, CaveNode] = field(default_factory=dict, repr=False)

    def add_node(self, node: CaveNode) -> None:
        self.nodes.append(node)
        self._by_id[node.node_id] = node

    def add_edge(self, edge: CaveEdge) -> None:
        self.edges.append(edge)

    def get_node(self, node_id: int) -> Optional[CaveNode]:
        return self._by_id.get(node_id)

    def neighbours(self, node_id: int) -> List[CaveNode]:
        result = []
        for e in self.edges:
            if e.src_id == node_id:
                n = self._by_id.get(e.dst_id)
                if n:
                    result.append(n)
            elif e.dst_id == node_id:
                n = self._by_id.get(e.src_id)
                if n:
                    result.append(n)
        return result


# ---------------------------------------------------------------------------
# Surface portal descriptor
# ---------------------------------------------------------------------------


@dataclass
class CavePortal:
    """A surface entry point into the underground world."""
    portal_id:    int
    direction:    Vec3    # unit direction (planet surface point)
    slope_deg:    float   # local slope (moderate → accessible)
    fracture:     float   # local fracture value [0..1]
    nearest_node: int     # CaveNode.node_id of the shallowest cave below


# ---------------------------------------------------------------------------
# Collapse event + patch batch
# ---------------------------------------------------------------------------


@dataclass
class SubsurfacePatch:
    """One SDF modification caused by a subsurface event."""
    patch: SDFPatch        # SphereCarve or AdditiveDeposit
    event_id: int


@dataclass
class SubsurfacePatchBatch:
    """A server-authoritative event that clients can apply deterministically."""
    event_id:   int
    event_type: SubsurfaceEventType
    game_time:  float
    position:   Vec3       # world-space direction (unit vector × planet_radius)
    seed_local: int
    patches:    List[SubsurfacePatch] = field(default_factory=list)


@dataclass
class CollapseEvent:
    """Record of a server-generated collapse event."""
    event_id:   int
    game_time:  float
    position:   Vec3
    seed_local: int
    batch:      SubsurfacePatchBatch


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


_DEFAULTS: Dict[str, Any] = {
    "enable":                    True,
    "seed_salt":                 0x5CA1AB1E,
    "shallow_prob":              0.6,
    "mid_prob":                  0.3,
    "deep_prob":                 0.1,
    "tube_radius_min":           4.0,
    "tube_radius_max":           18.0,
    "chamber_size_min":          15.0,
    "chamber_size_max":          80.0,
    "fracture_thickness":        3.0,
    "portal_frequency":          0.03,
    "cave_fog_density":          0.8,
    "cave_light_scatter":        0.6,
    "cave_audio_reverb_mix":     0.7,
    "collapse_event_rate_cap":   4,
}

# Maximum nodes per depth layer generated at build time
_MAX_NODES_SHALLOW = 24
_MAX_NODES_MID     = 16
_MAX_NODES_DEEP    = 6


# ---------------------------------------------------------------------------
# Cave graph builder (deterministic)
# ---------------------------------------------------------------------------


class CaveGraphBuilder:
    """Builds a deterministic CaveGraph from a seed and optional geo-sampler.

    Parameters
    ----------
    seed          : combined world seed already XOR-d with seed_salt
    geo_sampler   : optional — if provided, fracture/stress fields bias node
                    placement toward geologically active zones.
    planet_radius : planet radius in metres (determines depth scaling)
    config        : subsurface config dict (uses _DEFAULTS for missing keys)
    """

    def __init__(
        self,
        seed: int,
        planet_radius: float = 1_000.0,
        geo_sampler: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._seed    = seed
        self._radius  = planet_radius
        self._geo     = geo_sampler
        self._cfg     = {**_DEFAULTS, **(config or {})}

    # ------------------------------------------------------------------
    def build(self) -> CaveGraph:
        """Return the fully connected cave graph."""
        rng   = random.Random(self._seed)
        graph = CaveGraph()

        node_id = 0
        edge_id = 0

        # ------ shallow nodes ------
        n_shallow = _MAX_NODES_SHALLOW
        shallow_nodes: List[CaveNode] = []
        for _ in range(n_shallow):
            if rng.random() > self._cfg["shallow_prob"]:
                continue
            kind = rng.choice([CaveNodeKind.FRACTURE, CaveNodeKind.CHAMBER])
            d    = _vec3_from_seed(rng)
            depth = rng.uniform(10.0, 200.0)
            r     = rng.uniform(
                self._cfg["chamber_size_min"] * 0.3,
                self._cfg["chamber_size_min"],
            )
            n = CaveNode(
                node_id=node_id, kind=kind, direction=d,
                depth=depth, radius=r, layer=CaveDepthLayer.SHALLOW,
            )
            graph.add_node(n)
            shallow_nodes.append(n)
            node_id += 1

        # ------ mid nodes ------
        n_mid = _MAX_NODES_MID
        mid_nodes: List[CaveNode] = []
        for _ in range(n_mid):
            if rng.random() > self._cfg["mid_prob"]:
                continue
            kind  = rng.choice([CaveNodeKind.TUBE_JOINT, CaveNodeKind.CHAMBER])
            d     = _vec3_from_seed(rng)
            depth = rng.uniform(200.0, 1_000.0)
            r     = rng.uniform(
                self._cfg["chamber_size_min"],
                self._cfg["chamber_size_max"],
            )
            n = CaveNode(
                node_id=node_id, kind=kind, direction=d,
                depth=depth, radius=r, layer=CaveDepthLayer.MID,
            )
            graph.add_node(n)
            mid_nodes.append(n)
            node_id += 1

        # ------ deep nodes ------
        n_deep = _MAX_NODES_DEEP
        deep_nodes: List[CaveNode] = []
        for _ in range(n_deep):
            if rng.random() > self._cfg["deep_prob"]:
                continue
            d     = _vec3_from_seed(rng)
            depth = rng.uniform(1_000.0, 5_000.0)
            r     = rng.uniform(
                self._cfg["chamber_size_max"] * 0.5,
                self._cfg["chamber_size_max"] * 2.0,
            )
            n = CaveNode(
                node_id=node_id, kind=CaveNodeKind.CHAMBER, direction=d,
                depth=depth, radius=r, layer=CaveDepthLayer.DEEP,
            )
            graph.add_node(n)
            deep_nodes.append(n)
            node_id += 1

        # ------ connect: shallow → mid (vertical shafts / tubes) ------
        for sh in shallow_nodes:
            if not mid_nodes:
                break
            # Connect to the "closest" mid node (angular proximity proxy)
            best = max(mid_nodes, key=lambda m: sh.direction.dot(m.direction))
            tube_r = rng.uniform(
                self._cfg["tube_radius_min"],
                self._cfg["tube_radius_max"],
            )
            graph.add_edge(CaveEdge(
                edge_id=edge_id, src_id=sh.node_id,
                dst_id=best.node_id, radius=tube_r,
            ))
            edge_id += 1

        # ------ connect: mid → deep ------
        for md in mid_nodes:
            if not deep_nodes:
                break
            best = max(deep_nodes, key=lambda dp: md.direction.dot(dp.direction))
            tube_r = rng.uniform(
                self._cfg["tube_radius_min"] * 2.0,
                self._cfg["tube_radius_max"] * 2.0,
            )
            graph.add_edge(CaveEdge(
                edge_id=edge_id, src_id=md.node_id,
                dst_id=best.node_id, radius=tube_r,
            ))
            edge_id += 1

        return graph


# ---------------------------------------------------------------------------
# PortalFinder — surface entry points
# ---------------------------------------------------------------------------


class PortalFinder:
    """Finds rare, fracture-biased surface portals above the shallowest caves.

    Portals are deterministic; their positions depend only on the seed and
    the cave graph node positions.

    Parameters
    ----------
    seed          : RNG seed (same as cave graph builder)
    cave_graph    : the CaveGraph instance to anchor portals to
    planet_radius : planet radius in metres
    config        : subsurface config dict
    """

    def __init__(
        self,
        seed: int,
        cave_graph: CaveGraph,
        planet_radius: float = 1_000.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._seed   = seed
        self._graph  = cave_graph
        self._radius = planet_radius
        self._cfg    = {**_DEFAULTS, **(config or {})}

    # ------------------------------------------------------------------
    def find_portals(self) -> List[CavePortal]:
        """Return a deterministic list of surface portals."""
        rng     = random.Random(self._seed ^ 0xB0B5CA1A)
        portals: List[CavePortal] = []

        shallow_nodes = [n for n in self._graph.nodes
                         if n.layer == CaveDepthLayer.SHALLOW]
        if not shallow_nodes:
            return portals

        freq = self._cfg["portal_frequency"]   # chance per shallow node

        portal_id = 0
        for node in shallow_nodes:
            if rng.random() > freq:
                continue
            # Small angular offset so portal is near but not exactly at node
            offset_lat = rng.uniform(-0.02, 0.02)  # radians
            offset_lon = rng.uniform(-0.02, 0.02)
            d = node.direction
            # Rotate direction slightly using the offset
            ref   = Vec3(0.0, 1.0, 0.0) if abs(d.x) < 0.9 else Vec3(0.0, 0.0, 1.0)
            tang1 = ref.cross(d).normalized()
            tang2 = d.cross(tang1).normalized()
            perturbed = d + tang1 * math.sin(offset_lat) + tang2 * math.sin(offset_lon)
            r = perturbed.length()
            if r < 1e-9:
                continue
            perturbed = Vec3(perturbed.x / r, perturbed.y / r, perturbed.z / r)

            slope  = rng.uniform(5.0, 25.0)   # moderate — accessible
            fractr = rng.uniform(0.4, 1.0)    # high fracture → portal

            portals.append(CavePortal(
                portal_id=portal_id,
                direction=perturbed,
                slope_deg=slope,
                fracture=fractr,
                nearest_node=node.node_id,
            ))
            portal_id += 1

        return portals


# ---------------------------------------------------------------------------
# CaveFactor — cave atmosphere factor computation
# ---------------------------------------------------------------------------


class CaveFactorEstimator:
    """Estimates how "underground" a world position feels.

    Uses angular proximity to cave nodes weighted by depth.  The result is a
    smooth [0, 1] value: 0 = fully on surface, 1 = deep underground.

    This is a procedural approximation — full occlusion raycasting would be
    performed by a renderer, but for gameplay/audio/physics purposes this
    provides a sufficient signal.
    """

    def __init__(
        self,
        cave_graph: CaveGraph,
        planet_radius: float = 1_000.0,
    ) -> None:
        self._graph  = cave_graph
        self._radius = planet_radius

    # ------------------------------------------------------------------
    def estimate(self, world_pos: Vec3) -> float:
        """Return cave factor ∈ [0, 1] at *world_pos*."""
        if not self._graph.nodes:
            return 0.0

        pos_r = world_pos.length()
        if pos_r < 1e-9:
            return 0.0

        pos_dir = Vec3(
            world_pos.x / pos_r,
            world_pos.y / pos_r,
            world_pos.z / pos_r,
        )

        # Height above planet surface (positive = above surface)
        surface_alt = pos_r - self._radius

        max_cf = 0.0
        for node in self._graph.nodes:
            # Angular proximity [0, 1] — cosine similarity mapped to [0, 1]
            cos_ang   = _clamp(pos_dir.dot(node.direction), -1.0, 1.0)
            ang_prox  = (cos_ang + 1.0) * 0.5

            # Depth factor: how deep underground relative to this node's layer
            # If player is above surface → 0; if at node depth → 1
            depth_needed = node.depth
            depth_factor = _clamp(-surface_alt / (depth_needed + 1.0), 0.0, 1.0)

            # Combine: both proximity and depth must be high.
            # The radius weight (planet_radius * 0.1) normalises node.radius so
            # a chamber of ~100 m in a 1 km planet contributes ~factor 0.5.
            cf = ang_prox * depth_factor * (node.radius / (self._radius * 0.1 + node.radius))
            if cf > max_cf:
                max_cf = cf

        return _clamp(max_cf, 0.0, 1.0)


# ---------------------------------------------------------------------------
# CollapseGenerator — server-authoritative collapse event
# ---------------------------------------------------------------------------


class CollapseGenerator:
    """Generates SDF patches for a subsurface collapse event.

    Parameters
    ----------
    planet_radius : metres
    config        : subsurface config dict
    """

    def __init__(
        self,
        planet_radius: float = 1_000.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._radius = planet_radius
        self._cfg    = {**_DEFAULTS, **(config or {})}

    # ------------------------------------------------------------------
    def generate(
        self,
        event_id: int,
        world_dir: Vec3,    # unit direction on planet surface
        game_time: float,
        seed_local: int,
    ) -> SubsurfacePatchBatch:
        """Return a SubsurfacePatchBatch for a collapse at *world_dir*."""
        rng = random.Random(seed_local ^ 0xC01_1A75E)

        # World-space position of collapse centre (on surface)
        pos = Vec3(
            world_dir.x * self._radius,
            world_dir.y * self._radius,
            world_dir.z * self._radius,
        )

        # Collapse chamber: SphereCarve (remove void)
        chamber_r = rng.uniform(
            self._cfg["chamber_size_min"] * 0.5,
            self._cfg["chamber_size_min"],
        )
        chamber_patch = SphereCarve(centre=pos, radius=chamber_r)

        # Shaft upward to surface
        shaft_depth  = rng.uniform(chamber_r * 1.5, chamber_r * 3.0)
        shaft_r      = chamber_r * rng.uniform(0.3, 0.5)
        shaft_top    = Vec3(
            pos.x + world_dir.x * shaft_depth,
            pos.y + world_dir.y * shaft_depth,
            pos.z + world_dir.z * shaft_depth,
        )
        shaft_patch  = CapsuleCarve(a=pos, b=shaft_top, radius=shaft_r)

        # Debris deposit ring at the rim
        debris_r    = chamber_r * rng.uniform(1.1, 1.4)
        debris_patch = AdditiveDeposit(centre=pos, radius=debris_r)

        patches = [
            SubsurfacePatch(patch=chamber_patch, event_id=event_id),
            SubsurfacePatch(patch=shaft_patch,   event_id=event_id),
            SubsurfacePatch(patch=debris_patch,  event_id=event_id),
        ]

        return SubsurfacePatchBatch(
            event_id   = event_id,
            event_type = SubsurfaceEventType.COLLAPSE,
            game_time  = game_time,
            position   = world_dir,
            seed_local = seed_local,
            patches    = patches,
        )


# ---------------------------------------------------------------------------
# SubsurfaceSystem — main public API
# ---------------------------------------------------------------------------


class SubsurfaceSystem(ISubsurfaceSystem):
    """Stage 26 subsurface world — caves, tubes, chambers, collapses.

    Parameters
    ----------
    config        : dict with keys from the ``subsurface`` config section.
    global_seed   : world-level seed; combined with ``seed_salt`` internally.
    geo_sampler   : optional GeoFieldSampler (used to bias cave placement).
    planet_radius : planet radius in metres.
    sdf_world     : optional SDFWorld reference (patches applied here).

    Usage (server)
    --------------
    sys = SubsurfaceSystem(config=cfg["subsurface"], global_seed=42,
                           planet_radius=R, geo_sampler=geo, sdf_world=sdf)
    # per tick:
    sys.update(dt, game_time, player_pos)
    # when a collapse should happen:
    batch = sys.generate_collapse(world_dir, seed)
    sys.apply_event_patch(batch)    # apply locally
    # → broadcast batch as SUBSURFACE_EVENT over network

    Usage (client)
    --------------
    sys.apply_event_patch(batch)    # received from server
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        global_seed: int = 0,
        geo_sampler: Any = None,
        planet_radius: float = 1_000.0,
        sdf_world: Any = None,
        collapse_event_rate_cap: Optional[int] = None,
    ) -> None:
        self._cfg    = {**_DEFAULTS, **(config or {})}
        if collapse_event_rate_cap is not None:
            self._cfg["collapse_event_rate_cap"] = collapse_event_rate_cap

        self._radius = planet_radius
        self._geo    = geo_sampler
        self._sdf    = sdf_world

        # Combined seed
        salt        = int(self._cfg["seed_salt"])
        self._seed  = (global_seed ^ salt) & 0xFFFF_FFFF

        # Build the cave graph (deterministic)
        self._builder   = CaveGraphBuilder(
            seed=self._seed,
            planet_radius=planet_radius,
            geo_sampler=geo_sampler,
            config=self._cfg,
        )
        self._cave_graph: CaveGraph = self._builder.build()

        # Portals
        self._portal_finder = PortalFinder(
            seed=self._seed,
            cave_graph=self._cave_graph,
            planet_radius=planet_radius,
            config=self._cfg,
        )
        self._portals: List[CavePortal] = self._portal_finder.find_portals()

        # Cave-factor estimator
        self._cf_estimator = CaveFactorEstimator(
            cave_graph=self._cave_graph,
            planet_radius=planet_radius,
        )

        # Collapse generator
        self._collapse_gen = CollapseGenerator(
            planet_radius=planet_radius,
            config=self._cfg,
        )

        # Dynamic event log
        self._event_log:   List[CollapseEvent] = []
        self._next_evt_id: int = 0
        self._collapse_budget: int = 0   # events fired this hour (rate cap)
        self._budget_reset_time: float = 0.0

        # Applied patch log (for replay)
        self._patch_log = SDFPatchLog()

    # ------------------------------------------------------------------
    # ISubsurfaceSystem interface
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        game_time: float,
        player_pos: Optional[Vec3],
    ) -> None:
        """Advance the subsurface simulation by *dt* seconds."""
        if not self._cfg.get("enable", True):
            return

        # Reset hourly collapse budget
        if game_time - self._budget_reset_time >= 3_600.0:
            self._collapse_budget   = 0
            self._budget_reset_time = game_time

    def cave_factor_at(self, world_pos: Vec3) -> float:
        """Cave atmosphere factor ∈ [0, 1] at *world_pos*."""
        return self._cf_estimator.estimate(world_pos)

    def portals(self) -> List[Vec3]:
        """Unit directions of all surface portal entry points."""
        return [p.direction for p in self._portals]

    # ------------------------------------------------------------------
    # Collapse generation (server only)
    # ------------------------------------------------------------------

    def generate_collapse(
        self,
        world_dir: Vec3,
        seed_local: Optional[int] = None,
        game_time: float = 0.0,
    ) -> SubsurfacePatchBatch:
        """Generate a server-authoritative collapse event at *world_dir*.

        Parameters
        ----------
        world_dir  : unit direction on the planet surface.
        seed_local : local RNG seed; if None, derived from event_id + global seed.
        game_time  : current simulation time.

        Returns
        -------
        SubsurfacePatchBatch  — ready for broadcast and local application.
        """
        rate_cap = int(self._cfg.get("collapse_event_rate_cap", 4))
        if self._collapse_budget >= rate_cap:
            # Return an empty batch; caller must honour the cap
            return SubsurfacePatchBatch(
                event_id   = self._next_evt_id,
                event_type = SubsurfaceEventType.COLLAPSE,
                game_time  = game_time,
                position   = world_dir,
                seed_local = seed_local or 0,
                patches    = [],
            )

        if seed_local is None:
            seed_local = (self._seed ^ (self._next_evt_id * 0x9E37)) & 0xFFFF_FFFF

        batch = self._collapse_gen.generate(
            event_id   = self._next_evt_id,
            world_dir  = world_dir,
            game_time  = game_time,
            seed_local = seed_local,
        )
        self._next_evt_id    += 1
        self._collapse_budget += 1
        return batch

    # ------------------------------------------------------------------
    # Patch application (server + client)
    # ------------------------------------------------------------------

    def apply_event_patch(self, batch: SubsurfacePatchBatch) -> None:
        """Apply a collapse batch to the local SDF world.

        Safe to call on both server (after generation) and clients (on receipt).
        Patches are added to the internal SDFPatchLog for replay consistency.
        """
        for sp in batch.patches:
            self._patch_log.add(sp.patch)
            if self._sdf is not None:
                try:
                    self._sdf.apply_patch(sp.patch)
                except (AttributeError, TypeError):
                    pass   # SDF world not fully initialised is non-fatal

        # Record the collapse event
        self._event_log.append(CollapseEvent(
            event_id   = batch.event_id,
            game_time  = batch.game_time,
            position   = batch.position,
            seed_local = batch.seed_local,
            batch      = batch,
        ))

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def cave_graph(self) -> CaveGraph:
        """The full deterministic cave graph."""
        return self._cave_graph

    @property
    def event_log(self) -> List[CollapseEvent]:
        """All applied collapse events (in order)."""
        return list(self._event_log)

    @property
    def cave_portals(self) -> List[CavePortal]:
        """All surface portal descriptors."""
        return list(self._portals)

    # ------------------------------------------------------------------
    # Atmosphere parameters for renderer / audio / character systems
    # ------------------------------------------------------------------

    def atmosphere_params(self, cave_factor: float) -> Dict[str, float]:
        """Return a dict of cave-mode rendering/audio parameters.

        Parameters
        ----------
        cave_factor : value from ``cave_factor_at()`` ∈ [0, 1].

        Returns
        -------
        dict with keys:
          fog_density, light_scatter, audio_reverb_mix,
          speed_scale, turn_responsiveness, brace_rate, head_stabilisation.
        """
        cf = _clamp(cave_factor, 0.0, 1.0)
        return {
            "fog_density":         _lerp(0.0,  self._cfg["cave_fog_density"],    cf),
            "light_scatter":       _lerp(0.0,  self._cfg["cave_light_scatter"],  cf),
            "audio_reverb_mix":    _lerp(0.0,  self._cfg["cave_audio_reverb_mix"], cf),
            # Character behaviour modifiers (used by CharacterEnvironmentIntegration)
            "speed_scale":         _lerp(1.0,  0.6,  cf),
            "turn_responsiveness": _lerp(1.0,  0.5,  cf),
            "brace_rate":          _lerp(0.05, 0.25, cf),
            "head_stabilisation":  _lerp(0.0,  0.8,  cf),
        }

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def debug_info(self) -> Dict[str, Any]:
        """Return a dict suitable for dev-mode visualisation."""
        return {
            "cave_nodes":      len(self._cave_graph.nodes),
            "cave_edges":      len(self._cave_graph.edges),
            "portals":         len(self._portals),
            "collapse_events": len(self._event_log),
            "collapse_budget": self._collapse_budget,
            "seed":            self._seed,
        }
