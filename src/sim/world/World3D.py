"""World3D — server-authoritative 3D world core.

This is the single source of truth for:
  * planet geometry (SDFVolume)
  * global fields (FieldSet)
  * material registry (MaterialDB)
  * player entities (pos / vel / orient / shell)
  * body constraint graphs (BodyConstraintGraph per player)

World3D.tick(dt) advances the simulation one step:
  1. Advance simTime.
  2. Update field parameters if needed (only bumps fields_revision).
  3. Apply player intents (simple kinematics: pos += vel * dt).
  4. Clamp players to planet surface ± hover distance.
  5. Occasionally generate a cosmetic footprint SDFPatch under each player.
  6. Collect pending patches and deltas for network replication.

Public API
----------
World3D(seed, planet_radius)
  .tick(dt, player_intents)    → list[SDFPatch]  new patches this tick
  .add_player(player_id)       → None
  .remove_player(player_id)    → None
  .set_intent(player_id, intent) → None
  .get_player_state(player_id) → dict | None
  .all_player_states()         → list[dict]
  .sdf_volume                  → SDFVolume
  .field_set                   → FieldSet
  .material_db                 → MaterialDB
  .get_body(player_id)         → BodyConstraintGraph | None
  .sim_time                    → float
  .seed                        → int
  .to_baseline_dict()          → dict  (WORLD_BASELINE payload)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.sim.body.BodyConstraintGraph import BodyConstraintGraph
from src.sim.fields.FieldSet import FieldSet
from src.sim.materials.MaterialDB import MaterialDB
from src.sim.sdf.SDFPatch import SDFPatch, KIND_SPHERE_DENT
from src.sim.sdf.SDFVolume import SDFVolume


# ---------------------------------------------------------------------------
# Player record
# ---------------------------------------------------------------------------

class _PlayerRecord:
    __slots__ = (
        "player_id", "pos", "vel", "orient",
        "intent", "body", "_footprint_accumulator",
    )

    def __init__(self, player_id: str, spawn_pos: List[float]) -> None:
        self.player_id            = player_id
        self.pos:    List[float]  = list(spawn_pos)
        self.vel:    List[float]  = [0.0, 0.0, 0.0]
        self.orient: List[float]  = [0.0, 0.0, 0.0, 1.0]  # quaternion xyzw
        self.intent: Dict         = {}
        self.body                 = BodyConstraintGraph(player_id=player_id)
        self._footprint_accumulator: float = 0.0  # time since last patch

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "id":     self.player_id,
            "pos":    list(self.pos),
            "vel":    list(self.vel),
            "orient": list(self.orient),
        }


# ---------------------------------------------------------------------------
# World3D
# ---------------------------------------------------------------------------

_FOOTPRINT_INTERVAL  = 2.0   # seconds between cosmetic footprint patches
_FOOTPRINT_RADIUS    = 0.5   # patch radius (simulation units)
_FOOTPRINT_STRENGTH  = 0.08  # depth of footprint dent
_SURFACE_HOVER       = 1.8   # how far above the SDF surface a player floats
_MAX_MOVE_SPEED      = 12.0  # simulation units / second


class World3D:
    """Server-authoritative 3D world.

    Parameters
    ----------
    seed :
        World seed.
    planet_radius :
        Base planet radius in simulation units.
    """

    def __init__(self, seed: int = 42, planet_radius: float = 1000.0) -> None:
        self._seed          = int(seed)
        self._planet_radius = float(planet_radius)
        self._sim_time      = 0.0
        self._patch_id_seq  = 0

        self._sdf_volume = SDFVolume(radius=planet_radius)
        self._field_set  = FieldSet(planet_radius=planet_radius, seed=seed)
        self._material_db = MaterialDB(seed=seed)

        self._players: Dict[str, _PlayerRecord] = {}

        # Active zone centre (average of all player positions)
        self._active_zone_centre: List[float] = [0.0, planet_radius + _SURFACE_HOVER, 0.0]
        self._active_zone_radius: float       = 200.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def sdf_volume(self) -> SDFVolume:
        return self._sdf_volume

    @property
    def field_set(self) -> FieldSet:
        return self._field_set

    @property
    def material_db(self) -> MaterialDB:
        return self._material_db

    # ------------------------------------------------------------------
    # Player management
    # ------------------------------------------------------------------

    def add_player(self, player_id: str) -> None:
        if player_id in self._players:
            return
        spawn = self._default_spawn(player_id)
        self._players[player_id] = _PlayerRecord(player_id=player_id,
                                                  spawn_pos=spawn)

    def remove_player(self, player_id: str) -> None:
        self._players.pop(player_id, None)

    def set_intent(self, player_id: str, intent: Dict[str, Any]) -> None:
        rec = self._players.get(player_id)
        if rec is not None:
            rec.intent = dict(intent)

    def get_player_state(self, player_id: str) -> Optional[Dict[str, Any]]:
        rec = self._players.get(player_id)
        return rec.to_state_dict() if rec else None

    def all_player_states(self) -> List[Dict[str, Any]]:
        return [r.to_state_dict() for r in self._players.values()]

    def get_body(self, player_id: str) -> Optional[BodyConstraintGraph]:
        rec = self._players.get(player_id)
        return rec.body if rec else None

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(
        self,
        dt: float,
        player_intents: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[SDFPatch]:
        """Advance simulation by *dt* seconds.

        Parameters
        ----------
        dt :
            Time step in seconds.  Clamped to [0, 0.1].
        player_intents :
            ``{player_id: intent_dict}`` override (optional; otherwise uses
            last stored intent per player).

        Returns
        -------
        list[SDFPatch]
            New patches generated during this tick (may be empty).
        """
        dt = max(0.0, min(dt, 0.1))
        self._sim_time += dt

        if player_intents:
            for pid, intent in player_intents.items():
                self.set_intent(pid, intent)

        new_patches: List[SDFPatch] = []

        for rec in self._players.values():
            self._tick_player(rec, dt)
            patch = self._maybe_footprint(rec, dt)
            if patch is not None:
                self._sdf_volume.apply_patch(patch)
                new_patches.append(patch)

        self._update_active_zone()

        return new_patches

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tick_player(self, rec: _PlayerRecord, dt: float) -> None:
        intent = rec.intent
        move_x = float(intent.get("move_x", 0.0))
        move_z = float(intent.get("move_z", 0.0))
        yaw    = float(intent.get("look_yaw", 0.0))

        # Convert intent to world-space velocity in the tangent plane
        sin_y  = math.sin(yaw)
        cos_y  = math.cos(yaw)
        fwd_x  = sin_y
        fwd_z  = cos_y

        # Tangent-space move: forward = (sin_yaw, 0, cos_yaw), right = (cos_yaw, 0, -sin_yaw)
        dx = (fwd_x * move_z - cos_y * move_x) * _MAX_MOVE_SPEED
        dz = (fwd_z * move_z + sin_y * move_x) * _MAX_MOVE_SPEED

        # Clamp speed
        speed_sq = dx * dx + dz * dz
        if speed_sq > _MAX_MOVE_SPEED * _MAX_MOVE_SPEED:
            scale = _MAX_MOVE_SPEED / math.sqrt(speed_sq)
            dx *= scale
            dz *= scale

        rec.vel[0] = dx
        rec.vel[2] = dz

        rec.pos[0] += dx * dt
        rec.pos[2] += dz * dt

        # Keep player on planet surface
        self._clamp_to_surface(rec)

    def _clamp_to_surface(self, rec: _PlayerRecord) -> None:
        """Project player onto the SDF surface + hover distance."""
        x, y, z = rec.pos[0], rec.pos[1], rec.pos[2]
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1e-6:
            # At origin — push to north pole
            rec.pos[1] = self._planet_radius + _SURFACE_HOVER
            return
        # Normalise to surface
        nr  = (self._planet_radius + _SURFACE_HOVER) / r
        rec.pos[0] = x * nr
        rec.pos[1] = y * nr
        rec.pos[2] = z * nr

    def _maybe_footprint(
        self,
        rec: _PlayerRecord,
        dt: float,
    ) -> Optional[SDFPatch]:
        """Return a cosmetic SDFPatch if it is time to stamp a footprint."""
        rec._footprint_accumulator += dt
        if rec._footprint_accumulator < _FOOTPRINT_INTERVAL:
            return None
        rec._footprint_accumulator = 0.0

        self._patch_id_seq += 1
        # Place patch at the surface point directly below the player
        x, y, z = rec.pos[0], rec.pos[1], rec.pos[2]
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1e-6:
            return None
        scale = self._planet_radius / r
        return SDFPatch(
            patch_id = self._patch_id_seq,
            revision = self._sdf_volume.sdf_revision + 1,
            cx       = x * scale,
            cy       = y * scale,
            cz       = z * scale,
            radius   = _FOOTPRINT_RADIUS,
            strength = _FOOTPRINT_STRENGTH,
            kind     = KIND_SPHERE_DENT,
        )

    def _default_spawn(self, player_id: str) -> List[float]:
        """Deterministic spawn position on the north-pole-adjacent surface."""
        # Hash the player_id to get a small angular offset
        h     = hash(player_id) % 360
        angle = math.radians(h)
        tilt  = angle * 0.05   # small polar offset so players spread slightly
        r     = self._planet_radius + _SURFACE_HOVER
        return [
            r * math.sin(angle) * 0.05,
            r * math.cos(tilt),
            r * math.sin(angle) * 0.05,
        ]

    def _update_active_zone(self) -> None:
        players = list(self._players.values())
        if not players:
            return
        cx = sum(p.pos[0] for p in players) / len(players)
        cy = sum(p.pos[1] for p in players) / len(players)
        cz = sum(p.pos[2] for p in players) / len(players)
        self._active_zone_centre = [cx, cy, cz]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_baseline_dict(self) -> Dict[str, Any]:
        """Return WORLD_BASELINE payload."""
        return {
            "seed":             self._seed,
            "planet_radius":    self._planet_radius,
            "sim_time":         self._sim_time,
            "sdf_revision":     self._sdf_volume.sdf_revision,
            "fields_revision":  self._field_set.fields_revision,
            "active_zone":      {
                "centre": self._active_zone_centre,
                "radius": self._active_zone_radius,
            },
        }
