# DUST — Technical Constraints

This document defines what is and is not permitted in the technical implementation. These are hard constraints, not preferences. Any code that violates these constraints must be refactored before merging.

---

## 1. Server Authority

### Required
- All simulation state (terrain SDF, material fields, climate fields, astro positions, character positions and shell occupancy) is owned and computed by the server.
- Clients send only `PlayerIntent` (direction + magnitude, camera direction). Clients do not compute simulation outcomes.
- The server broadcasts authoritative state snapshots to clients at a fixed rate.
- Clients are permitted to interpolate between two already-received snapshots for smooth rendering (backward-looking interpolation only). Clients must not predict or extrapolate simulation state beyond the most recently received snapshot.

### Forbidden
- Any simulation computation (GeoEvents, ShellOccupancy, MaterialOperators, CharacterForces) running on the client.
- Client-side terrain modification.
- Client-side character physics resolution.

---

## 2. Determinism

### Required
- The entire world must be reproducible from a single integer seed on the same build.
- A fixed tick rate governs all simulation updates; floating-point operations must be order-consistent across ticks.
- Snapshots must be serialisable and replayable: given the same seed and the same sequence of `PlayerIntent` inputs, the simulation must produce byte-identical state at every tick.

### This applies to:
- Terrain generation (SDF initial state)
- Material noise patterns
- Astronomical parameters (within config ranges)
- Fault network topology and stress thresholds
- Geo-event threshold randomisation
- Genesis accelerated pass

---

## 3. Texture Prohibition

### Forbidden
- Loading any bitmap image from disk as an albedo, normal, roughness, metallic, height, or emissive texture.
- Using any runtime texture atlas, sprite sheet, or UV-mapped image.
- Referencing any `.png`, `.jpg`, `.tga`, `.dds`, `.exr`, `.hdr`, or equivalent image file as a material input.

### Permitted
- GPU render textures generated entirely at runtime from code.
- Framebuffer / render-target textures used as post-process intermediaries.
- Procedurally generated 2D fields computed on CPU or GPU (noise, LUT computed from math, etc.).

---

## 4. External Asset Prohibition

### Forbidden
- Any third-party asset package, asset store purchase, or externally authored model.
- Any audio file sourced from outside the project's procedural audio pipeline.
- Any SDK, middleware, or plugin that is not part of the base engine (e.g. no Speedtree, no Houdini bridge, no third-party physics add-on, no third-party audio middleware).
- Any pre-authored animation clip (`.anim`, `.fbx`, `.bvh`, or equivalent).
- Any pre-authored skeleton mesh for the player character.

### Permitted
- Standard engine packages that ship as part of the target engine's core distribution.
- Engine built-in procedural noise nodes / shader graph nodes.
- Custom code written within this repository.

---

## 5. Material System

All surface and shell materials must be computed from MaterialState fields:

| Field | Description |
|-------|-------------|
| `grain_size` | Physical grain size of the material |
| `porosity` | Void fraction |
| `compaction` | Degree of compaction |
| `cementation` | Degree of cementation |
| `fracture` | Accumulated fracture damage |
| `temp` | Temperature |
| `melt_fraction` | Fraction in liquid/melt state |
| `wetness` | Water / humidity saturation |
| `layer` | `film` / `loose` / `substrate` |
| `noise_octaves` | Multi-octave fractal noise (seed-reproducible) |

The `state → physics` resolver must produce: density, friction, strength, viscosity, yield, liftability, settle rate, and acoustic profile. No other data sources are permitted as material physics inputs.

---

## 6. Geometry System

- **SDF-voxels**: terrain is represented as a signed distance field on a voxel grid.
- **Smoothed mesh**: the rendered surface is extracted from the SDF via a smoothed isosurface algorithm (e.g. dual contouring or smooth marching cubes).
- **No pre-authored meshes**: the planet surface, caves, and all geological features are generated from SDF operations only.
- **Dynamic modification**: the SDF must support runtime patching (GeoEvents modify it; patches are stored as deltas).
- **Shell geometry**: the player shell is derived from skeleton slot envelopes filled with voxelised material, not from a pre-authored character mesh.

---

## 7. World State Storage

- World modifications (SDFEdits from GeoEvents, RuptureEvents, ImpactEvents) must be stored as **delta patches**, not as full SDF snapshots.
- A delta patch contains: world-space origin, affected radius, operation type, magnitude.
- The list of applied patches must be serialisable to JSON.
- Total save data for a session must be proportional to the number of events that occurred, not to world size.
- Server snapshots for replay must include: tick number, seed, all applied delta patches, and all PlayerIntent sequences.

---

## 8. Admin Interface Isolation

- The admin panel is accessible only to the session host and must never appear in the simulation view presented to players.
- Admin controls (genesis/reset, snapshot, tuning) must operate through the server's own API, not by directly writing client-visible state.
- Tuning profile changes must be allowlisted: only parameters defined in `CONFIG_DEFAULTS.json` may be modified at runtime.

---

## 9. Permitted Exceptions

The following are documented exceptions to the above rules. Each requires explicit justification in the commit that introduces it:

| Exception type | Condition for allowing |
|---------------|----------------------|
| Engine built-in skybox LUT | Only if procedural sky shader is not feasible in the engine; must be replaced before vertical slice |
| Debug/editor-only bitmap | Allowed in editor-only tools, must be stripped from runtime builds |
| Font texture atlas | Permitted only for debug/editor overlay, never in simulation view |
| Client-side interpolation | Permitted for smooth rendering between server snapshots; must not affect simulation state |

---

## 10. Validation

All constraints in this document must be reflected in `VALIDATION_RULES.md` as checkable rules. Every pull request that introduces a new asset or dependency must be reviewed against this document before merge.
