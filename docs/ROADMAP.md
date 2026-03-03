# DUST — Roadmap

Global development stages. Each stage builds on the previous. This list defines scope, not schedule.

---

| Stage | Title | Goal |
|-------|-------|------|
| 1 | **Conceptual Manifest & Style Constraints** | Fix the project contract: philosophy, visual rules, simulation boundaries, controls, technical prohibitions, default config. No code written. |
| 2 | **Server Architecture & Tick Loop** | Authoritative server: fixed tick, deterministic state machine, snapshot serialisation, client intent intake, interest management skeleton. |
| 3 | **Planetary Mathematics & Celestial System** | Orbital mechanics (binary suns, moon, ring), planet rotation, InsolationField, RingShadowField, TidalStress, ImpactFlux, coordinate system, reproducible seed pipeline. |
| 4 | **PlanetTileGrid, PlateMap & FaultGraph** | Spherical tile grid, tectonic plate assignment, plate velocity, boundary classification, fault segment network, stress accumulation, creep, rupture event generation. |
| 5 | **Terrain Generation & SDF** | SDF-based planet surface: procedural geology, elevation, craters, ridges, caves, smoothed mesh extraction, LOD. No pre-authored meshes. |
| 6 | **Material System** | MaterialState fields (grain size, porosity, compaction, temp, melt fraction, wetness, layer), MaterialOperator pipeline (melt/solidify/freeze/fracture/flow/lift), mass-conservation accounting. |
| 7 | **Climate & Atmospheric Systems** | InsolationField → Climate derivation (temperature, pressure, wind, dust, humidity), thermal inertia, snow accumulation and transport, lava flow operator. |
| 8 | **Pixel Render Pipeline** | Downscale/upscale stage, ordered dithering, palette quantisation, light banding, cinematic tone mapping, three-layer depth compositing. |
| 9 | **Player Entity: Core + Virtual Skeleton + Shell** | CoreSeed identity, constraint-graph skeleton (slots/frames/constraints/actuators), shell fill/loss/regen cycle, ShellOccupancy driven by CharacterForces and surface contact. |
| 10 | **Character Physics & Intent System** | CharacterController with intent-only inputs, CharacterForces from environment, reflex states (stumble/slide/grab/recover/brace), multi-contact climbing, R-rebirth. |
| 11 | **Geological Event System** | GeoRiskFields, threshold logic, GeoEvent spawning (rift/landslide/collapse/scree), SDFEdits delta patches, RuptureEvent, environmental precursor signals. |
| 12 | **Sky & Celestial Rendering** | Binary sun disks, eclipse compositing, ring projection, moon, ring shadow bands on terrain, procedural sky scattering. |
| 13 | **Audio — Procedural / Reactive** | Wind synthesis, geological event sounds, material acoustic profiles derived from MaterialState, footstep feedback, subsurface resonance, deterministic voice budget. |
| 14 | **Genesis & World Lifecycle** | Accelerated Genesis pass using same simulation laws, playability gates, repair/regenerate pipeline, world fail-state detection (runaway heat / tectonic lock / orbital divergence), Reset with snapshot archival. |
| 15 | **Admin Interface** | Host-only panel (outside simulation view): lifecycle controls (genesis/reset), snapshots, metrics/queues/mass drift, planet maps (heat/stress/impact/shadow/snow), trace/replay, tuning profile allowlist. |
| 16 | **Multiplayer** | Two-player server authoritative session: intent replication, snapshot broadcast, shell/skeleton delta replication, anchor/assist replication, geo-event parameter replication. |
| 17 | **Vertical Slice** | One contiguous region of the planet playable end-to-end: full pipeline from astro to shell, all systems active, Genesis PASS, two players can connect and walk, movement stable, shell fills/loses/flows honestly, snow/wind/faults manifest without scripts, R always resolves. |
| 18 | **Full Planet Traversal** | Spherical world traversal, LOD streaming, seamless day/night on sphere, ring shadow on sphere, full climate field on sphere. |
| 19 | **Polish & Certification** | Performance targets met, all validation rules pass, no prohibited assets, reproducibility verified, release candidate. |

---

## Notes

- Stage 1 (this stage) produces documentation only — no executable code.
- Stages 2–4 establish the server skeleton and planetary data structures before simulation content begins.
- Stages 5–7 may proceed partly in parallel once Stage 5 data structures are defined.
- Stage 17 is the first playable milestone against which all constraints are verified.
- Stages 18–19 are post-vertical-slice scope.
