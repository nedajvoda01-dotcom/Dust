# BARYCENTER — Technical Constraints

This document defines what is and is not permitted in the technical implementation. These are hard constraints, not preferences. Any code that violates these constraints must be refactored before merging.

---

## 1. Texture Prohibition

### Forbidden
- Loading any bitmap image from disk as an albedo, normal, roughness, metallic, height, or emissive texture.
- Using any runtime texture atlas, sprite sheet, or UV-mapped image.
- Referencing any `.png`, `.jpg`, `.tga`, `.dds`, `.exr`, `.hdr`, or equivalent image file as a material input.

### Permitted
- GPU render textures generated entirely at runtime from code.
- Framebuffer / render-target textures used as post-process intermediaries.
- Procedurally generated 2D fields computed on CPU or GPU (noise, LUT computed from math, etc.).

---

## 2. External Asset Prohibition

### Forbidden
- Any third-party asset package, asset store purchase, or externally authored model.
- Any audio file sourced from outside the project's procedural audio pipeline.
- Any SDK, middleware, or plugin that is not part of the base engine (e.g. no Speedtree, no Houdini bridge, no third-party physics add-on, no third-party audio middleware).
- Any pre-authored animation clip (`.anim`, `.fbx`, `.bvh`, or equivalent).

### Permitted
- Standard engine packages that ship as part of the target engine's core distribution.
- Engine built-in procedural noise nodes / shader graph nodes.
- Custom code written within this repository.

---

## 3. Material System

All surface materials must be computed procedurally from:

| Input | Description |
|-------|-------------|
| `slope` | Angle of surface normal relative to gravity vector |
| `curvature` | Local mean curvature of SDF surface |
| `height` | Elevation above sea-level datum |
| `dust_accumulation` | Dust field value at this surface point |
| `wetness` | Humidity field value at this surface point |
| `noise_octaves` | Multi-octave fractal noise (seed-reproducible) |

No other data sources are permitted as material inputs.

---

## 4. Geometry System

- **SDF-voxels**: terrain is represented as a signed distance field on a voxel grid.
- **Smoothed mesh**: the rendered surface is extracted from the SDF via a smoothed isosurface algorithm (e.g. dual contouring or smooth marching cubes).
- **No pre-authored meshes**: the planet surface, caves, and all geological features are generated from SDF operations only.
- **Dynamic modification**: the SDF must support runtime patching (GeoEvents modify it; patches are stored as deltas).

---

## 5. Reproducibility Requirement

> **The entire world must be reproducible from a single integer seed.**

This applies to:
- Terrain generation (SDF initial state)
- Material noise patterns
- Astronomical parameters (within config ranges)
- Geo-event threshold randomisation

A given seed must produce byte-identical initial world state on any machine running the same build.

---

## 6. World State Storage

- World modifications (SDFEdits from GeoEvents) must be stored as **delta patches**, not as full SDF snapshots.
- A delta patch contains: world-space origin, affected radius, operation type, magnitude.
- The list of applied patches must be serialisable to JSON.
- Total save data for a session must be proportional to the number of events that occurred, not to world size.

---

## 7. Permitted Exceptions

The following are documented exceptions to the above rules. Each requires explicit justification in the commit that introduces it:

| Exception type | Condition for allowing |
|---------------|----------------------|
| Engine built-in skybox LUT | Only if procedural sky shader is not feasible in the engine; must be replaced before vertical slice |
| Debug/editor-only bitmap | Allowed in editor-only tools, must be stripped from runtime builds |
| Font texture atlas | Permitted only for debug/editor overlay, never in simulation view |

---

## 8. Validation

All constraints in this document must be reflected in `VALIDATION_RULES.md` as checkable rules. Every pull request that introduces a new asset or dependency must be reviewed against this document before merge.
