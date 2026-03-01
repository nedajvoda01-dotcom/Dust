# BARYCENTER — Roadmap

Global development stages. Each stage builds on the previous. This list defines scope, not schedule.

---

| Stage | Title | Goal |
|-------|-------|------|
| 1 | **Conceptual Manifest & Style Constraints** | Fix the project contract: philosophy, visual rules, simulation boundaries, controls, technical prohibitions, default config. No code written. |
| 2 | **Engine Core & Planetary Mathematics** | Implement orbital mechanics (binary suns, moon, ring), planet rotation, SDF grid foundation, coordinate system, reproducible seed pipeline. |
| 3 | **Terrain Generation** | SDF-based planet surface: procedural geology, elevation, craters, ridges, caves, smoothed mesh extraction, LOD. |
| 4 | **Atmospheric & Insolation Systems** | Insolation field from AstroSystem, climate field derivation (temperature, pressure, wind, dust, humidity), runtime atmospheric shader. |
| 5 | **Pixel Render Pipeline** | Downscale/upscale stage, ordered dithering, palette quantisation, light banding, cinematic tone mapping, three-layer depth compositing. |
| 6 | **Character Physics & Intent System** | Character controller with intent-only inputs, CharacterForces from environment, reflex states (stumble/slide/grab/recover/brace). |
| 7 | **Geological Event System** | GeoRiskFields, threshold logic, GeoEvent spawning (rift/landslide/collapse/scree), SDF delta patches, environmental precursor signals. |
| 8 | **Sky & Celestial Rendering** | Binary sun disks, eclipse compositing, ring projection, moon, ring shadow bands on terrain, procedural sky scattering. |
| 9 | **Audio — Procedural / Reactive** | Wind synthesis, geological event sounds, character footstep material feedback, subsurface resonance, no sampled audio files. |
| 10 | **Vertical Slice** | One contiguous region of the planet playable end-to-end: full pipeline from astro to character, all systems active, all visual rules met. |
| 11 | **Full Planet Traversal** | Spherical world traversal, LOD streaming, seamless day/night on sphere, ring shadow on sphere, full climate field on sphere. |
| 12 | **Polish & Certification** | Performance targets met, all validation rules pass, no prohibited assets, reproducibility verified, release candidate. |

---

## Notes

- Stage 1 (this stage) produces documentation only — no executable code.
- Stages 2–5 may proceed partly in parallel once Stage 2 data structures are defined.
- Stage 10 is the first playable milestone against which all constraints are verified.
- Stages 11–12 are post-vertical-slice scope.
