# DUST — Validation Rules

This document defines the rules that must be verifiable automatically (or by structured manual review) before any milestone merge. Rules marked `[AUTO]` are candidates for scripted CI checks. Rules marked `[REVIEW]` require human inspection.

---

## Category 1 — UI Absence

| # | Rule | Method |
|---|------|--------|
| 1.1 | No persistent screen-space UI layer exists in the simulation scene | `[REVIEW]` Scene graph inspection: zero UI canvas components active during play |
| 1.2 | No health / stamina / resource / shell-fill bar components exist in any scene | `[AUTO]` Static scan: no component types matching `HealthBar`, `StaminaBar`, `ResourceUI`, `HUD`, `ShellBar` |
| 1.3 | No waypoint, marker, or compass component exists in any scene | `[AUTO]` Static scan: no component types matching `Waypoint`, `Compass`, `Minimap`, `Marker` |
| 1.4 | No subtitle or narrative text overlay exists in simulation view | `[REVIEW]` Play-mode inspection |

---

## Category 2 — Texture / Asset Prohibition

| # | Rule | Method |
|---|------|--------|
| 2.1 | No bitmap image files loaded at runtime in simulation build | `[AUTO]` Build artifact scan: no `.png`, `.jpg`, `.tga`, `.dds`, `.exr`, `.hdr` references in runtime asset manifest |
| 2.2 | No external asset packages present in project | `[AUTO]` Dependency manifest scan: all packages are engine-core or internal |
| 2.3 | No pre-authored mesh files used for terrain, character environment, or character body | `[REVIEW]` Asset folder scan: terrain/environment/character folders contain no `.fbx`, `.obj`, `.gltf` files |
| 2.4 | No pre-authored animation clips or skeleton mesh used for character | `[REVIEW]` Asset folder scan: character folder contains no `.anim`, `.bvh`, `.fbx` files |

---

## Category 3 — Render Pipeline

| # | Rule | Method |
|---|------|--------|
| 3.1 | Every rendered frame passes through `PixelRenderStage` (downscale/dither/quantise) | `[REVIEW]` Render pipeline graph shows `PixelRenderStage` as non-bypassable step |
| 3.2 | Three depth layers (foreground/midground/background) are visually distinct in test frames | `[REVIEW]` Screenshot comparison against style reference: foreground darkest, background palest |
| 3.3 | Atmospheric haze produces visible gradient between midground and background | `[REVIEW]` Fog/haze parameter check: `atmosphere.haze_density > 0` and background layer luminance > midground luminance |
| 3.4 | Dithering is ordered (Bayer matrix), not random noise | `[REVIEW]` Shader code review: dither function uses matrix lookup, not `frac(sin(...))` random |

---

## Category 4 — Celestial Systems

| # | Rule | Method |
|---|------|--------|
| 4.1 | Two distinct sun disks are rendered on the sky | `[REVIEW]` Play-mode screenshot: two light sources visible on sky sphere |
| 4.2 | Ring is visible on sky with correct projection | `[REVIEW]` Play-mode screenshot: ring arc visible above horizon |
| 4.3 | Moon is rendered and orbits correctly | `[REVIEW]` Time-lapse test: moon position changes over `moon.period_minutes` |
| 4.4 | Ring shadow bands appear on terrain surface | `[REVIEW]` Play-mode screenshot at correct ring angle: shadow bands visible on ground |
| 4.5 | TidalStress field is non-zero and varies with moon position | `[AUTO]` Simulation test: tidal stress at surface points changes as moon orbits |
| 4.6 | ImpactFlux field is non-zero in ring-resonance zones | `[REVIEW]` PlanetTileGrid inspection: impact flux values positive in expected latitude bands |

---

## Category 5 — Controls / Intent System

| # | Rule | Method |
|---|------|--------|
| 5.1 | Player input is limited to: move direction, turn/look, camera pitch, rebirth, system pause | `[REVIEW]` Input system review: no additional action bindings exist |
| 5.2 | Character has at least four autonomous reflex states active | `[REVIEW]` CharacterController code review: `stumble`, `slide`, `grab`, `recover` states present |
| 5.3 | No "interact" or "action" input binding exists | `[AUTO]` Input map scan: no binding named `Interact`, `Action`, `Use`, `Jump` |
| 5.4 | Rebirth (`R`) relocates core to safe surface within config radius | `[AUTO]` Simulation test: rebirth call places character on traversable surface within `rebirth.max_radius` |
| 5.5 | Rebirth deposits prior shell mass at previous location | `[AUTO]` Mass accounting test: world mass before and after rebirth is conserved |

---

## Category 6 — Simulation Integrity

| # | Rule | Method |
|---|------|--------|
| 6.1 | Given identical seed and time step, world state is byte-identical across runs | `[AUTO]` Determinism test: run simulation N steps twice from same seed, compare output |
| 6.2 | GeoEvents can only be triggered by GeoRiskField threshold, not by player proximity | `[REVIEW]` GeoEvent spawner code review: no player-distance condition in spawn logic |
| 6.3 | SDF modifications are stored as delta patches, not full-field snapshots | `[AUTO]` Save file inspection: save data size proportional to event count, not world size |
| 6.4 | Player cannot directly modify terrain SDF (except cosmetic footprint depth) | `[REVIEW]` Code review: no direct SDF write calls from PlayerInput or CharacterController |
| 6.5 | Shell material obeys the same MaterialOperators as terrain material | `[REVIEW]` Code review: ShellOccupancy uses identical operator pipeline as terrain MaterialState |
| 6.6 | Mass is conserved across MaterialOperator transitions (shell ↔ terrain) | `[AUTO]` Mass accounting test: total material mass in world constant across operator ticks (within float tolerance) |
| 6.7 | FaultGraph accumulates stress and produces RuptureEvents independently of player position | `[REVIEW]` FaultGraph code review: no player proximity check in stress accumulation or rupture logic |

---

## Category 7 — Server Authority

| # | Rule | Method |
|---|------|--------|
| 7.1 | All simulation state is computed on the server; clients receive state snapshots only | `[REVIEW]` Architecture review: no simulation computation in client code paths |
| 7.2 | Client code contains no MaterialOperator, GeoEvent, or ShellOccupancy logic | `[AUTO]` Static scan: MaterialOperator / GeoEvent / ShellOccupancy class references absent from client module |
| 7.3 | Client-side interpolation does not affect server state | `[REVIEW]` Code review: client interpolation writes only to render state, never to simulation state |
| 7.4 | Admin panel is inaccessible to non-host clients | `[REVIEW]` Server session logic review: admin API endpoints gated by host identity check |

---

## Category 8 — Genesis & World Lifecycle

| # | Rule | Method |
|---|------|--------|
| 8.1 | Genesis uses the same MaterialOperators and simulation laws as the live simulation | `[REVIEW]` Genesis code review: Genesis pass calls the same operator pipeline |
| 8.2 | Genesis result passes playability gates before world is exposed to players | `[AUTO]` Genesis test: traversable zone count ≥ config minimum; runaway heat flag is false |
| 8.3 | World fail-state detection triggers server Reset, not crash | `[AUTO]` Simulation test: inject runaway heat condition; verify Reset is called and genesis reruns |
| 8.4 | Reset archives the final world state snapshot before launching new Genesis | `[REVIEW]` Reset code review: snapshot serialisation called before Genesis starts |

---

## Category 9 — Config Compliance

| # | Rule | Method |
|---|------|--------|
| 9.1 | All numerical parameters from `CONFIG_DEFAULTS.json` are used as initial values, including all AstroSystem, Climate, MaterialOperator, and pixel parameters | `[REVIEW]` Config loading code review: each key maps to a runtime parameter |
| 9.2 | No hardcoded magic numbers for astronomical, climate, pixel, or material parameters | `[AUTO]` Static analysis: flag float/int literals in AstroSystem, Climate, MaterialOperators, PixelRenderStage that are not read from config |
| 9.3 | Admin tuning changes are restricted to keys present in `CONFIG_DEFAULTS.json` | `[AUTO]` Admin API handler: unknown keys rejected with error response |

---

## Validation Cadence

- Rules 2.1, 2.2, 5.3, 6.1, 6.3, 6.6, 7.2, 9.3 — checked on every PR via CI.
- All `[REVIEW]` rules — checked at each stage milestone review.
- Full rule set — checked at Vertical Slice (Stage 17) and Release Candidate (Stage 19).
