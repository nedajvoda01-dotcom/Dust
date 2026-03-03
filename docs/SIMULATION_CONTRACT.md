# DUST — Simulation Contract

This document defines the closed causal loop that governs all dynamic behaviour in DUST. Every runtime event must be traceable to this graph. Anything not in this graph does not happen.

---

## Causal Graph

```
AstroSystem (2 stars + ring + moon)
  └─► PlanetSpin
  └─► BinarySunAngles
  └─► EclipseState
  └─► RingShadowField
  └─► TidalStress       (from moon)
  └─► ImpactFlux        (from ring/resonances)

BinarySunAngles + EclipseState
  └─► InsolationField   (surface energy field, per star)

RingShadowField
  └─► InsolationField   (shadow bands modulate field)

PlanetSpin
  └─► InsolationField   (day/night cycle)

InsolationField
  └─► Climate
        ├─► Temperature field
        ├─► Pressure field
        ├─► Wind field (magnitude + direction)
        ├─► Humidity field
        └─► Dust suspension field

TidalStress
  └─► FaultGraph        (accumulates tidal loading on fault segments)

Climate + TidalStress + ImpactFlux
  └─► GeoRiskFields
        ├─► stress           (thermal expansion / fault loading)
        ├─► slope_instability (erosion + freeze/thaw + dust loading)
        └─► void_risk        (subsurface pressure / dissolution)

FaultGraph (when stress + lock state → rupture threshold)
  └─► RuptureEvent
        ├─► StressFlow       (redistributes stress on adjacent segments)
        ├─► DamageFlow       (comminution and fracture of surface material)
        └─► SignalFlow       (infrasound / vibration emitted into world)

GeoRiskFields (when thresholds exceeded)
  └─► GeoEvents
        ├─► RiftEvent        (rare: fault opening, terrain split)
        ├─► LandslideEvent   (moderate: slope failure, debris flow)
        ├─► CollapseEvent    (rare: void collapse, sinkhole)
        └─► ScreeEvent       (frequent: small rolling debris)

ImpactFlux (when impact probability threshold crossed)
  └─► ImpactEvent
        ├─► SDFEdits         (crater formation)
        ├─► SignalFlow        (shockwave)
        └─► DustPlume        (ejecta → dust suspension)

GeoEvents + RuptureEvent + ImpactEvent
  └─► SDFEdits (permanent deformation of terrain SDF field)
        ├─► surface height modifications
        └─► cave/void modifications

MaterialOperators (applied every tick by the server)
  melt / solidify       (Crust ↔ Magma ↔ Glass)
  freeze / thaw         (Snow ↔ Ice ↔ IceFilm ↔ Water)
  fracture / comminute  (Rock → Debris → Regolith)
  compaction / cementation (loose → substrate → crust)
  lift / settle         (ground ↔ AerosolDust)
  flow                  (downhill transport of loose/liquid material)

Climate + SDFEdits + MaterialOperators
  └─► CharacterForces
        ├─► wind_push        (from Wind field)
        ├─► shell_drag_loss  (wind / acceleration strips shell material)
        ├─► friction         (from surface material state: slope/dust/ice/wet/lava)
        ├─► slope_gravity    (from terrain normal under character)
        └─► quake_impulse    (from RiftEvent / RuptureEvent / ImpactEvent)

CharacterForces + PlayerIntent
  └─► CharacterController
        ├─► balance_state    (upright / leaning / stumbling / fallen)
        ├─► reflex_state     (grabbing / crouching / sliding / recovering / brace)
        ├─► velocity         (resultant of intent + forces)
        └─► position         (integrated from velocity)

CharacterController + MaterialOperators
  └─► ShellOccupancy
        ├─► FillRate         (material absorbed from dominant contact surface)
        ├─► LossRate         (wind / acceleration / lava / impact)
        ├─► SlotMass         (per-slot current fill fraction)
        └─► ExposureState    (slot below threshold → degradation → regen)
```

---

## System Descriptions

### AstroSystem
Computes positions of both suns (binary star system), the moon, and the ring geometry relative to the planet at each simulation tick. Outputs:
- `sun_primary_angle` (azimuth, elevation)
- `sun_secondary_angle` (azimuth, elevation)
- `eclipse_fraction` (0–1, how much primary is occluded by secondary or moon)
- `moon_position` (world-space vector)
- `tidal_stress_vector` (tidal force from moon, per surface point)
- `impact_flux` (probability density of ring debris impacts, per surface area)

### PlanetTileGrid
Spherical grid of tiles covering the planet surface. Holds coarse global fields (insolation, temperature, pressure, wind, dust, tidal stress, impact flux) that are interpolated into local simulation zones. All global fields live here.

### PlateMap
Each tile is assigned to a tectonic plate. Plates carry a velocity vector on the sphere. Plate boundaries are classified: convergent, divergent, transform, or diffuse. Boundary classification feeds into FaultGraph and GeoRiskFields.

### FaultGraph
A network of fault segments on the planet surface. Each segment tracks:
- Accumulated stress (from tidal loading, thermal cycling, plate motion)
- Lock state (stuck / creeping / ruptured)
- Creep rate

When a segment reaches rupture threshold, a RuptureEvent is generated, which redistributes stress, comminutes surface material, and emits a SignalFlow.

### InsolationField
Combines sun angles, eclipse state, ring shadow, and day/night cycle into a surface energy field (W/m² equivalent in simulation units). This field is the single source of thermal truth for the Climate system.

### Climate
Derives temperature, pressure, wind, humidity, and dust fields from InsolationField. These fields evolve over time with inertia (thermal mass, momentum). They do not change instantly when InsolationField changes.

### MaterialState
Every unit of material carries a state record:
- `grain_size`, `porosity`, `compaction`, `cementation`, `fracture`
- `temp`, `melt_fraction`, `wetness`
- Layer: `film` / `loose` / `substrate`

MaterialOperators transform MaterialState each tick, conserving mass. The resolver `state → physics` outputs: density, friction, strength, viscosity, yield, liftability, settle rate, and acoustic profile.

### GeoRiskFields
Three scalar fields computed from Climate history and SDF terrain data:
- **stress**: accumulated thermal cycling load + tidal load
- **slope_instability**: erosion + overburden + saturation + fracture damage
- **void_risk**: subsurface pressure differentials

### GeoEvents
When any GeoRiskField exceeds its threshold (defined in config), a GeoEvent is spawned at that location. Events are not scripted. Thresholds may be randomised within config ranges using the world seed.

### SDFEdits
Permanent modifications to the terrain SDF field. Stored as delta patches (not full-field copies). Applied when GeoEvents or ImpactEvents resolve. Caves and voids are part of the SDF.

### CharacterForces
A set of per-frame force vectors accumulated from environment systems and applied to the character physics body. The player does not control these forces. They are consequences of location and time.

### CharacterController
Receives `PlayerIntent` (direction + magnitude of movement desire, camera direction) and `CharacterForces` (from environment). Computes final movement, with reflex states overriding intent when forces exceed balance thresholds.

### ShellOccupancy
Tracks fill fraction per skeleton slot. Each tick:
- FillRate: shell absorbs material from the dominant contact surface.
- LossRate: wind, acceleration, lava contact, impacts strip shell material.
- If `LossRate > FillRate` for sustained duration, slot degrades.
- A degraded slot is replaced by a new empty subgraph (regen); the slot is inactive until it refills.

### Genesis
The server runs an accelerated pass of the same simulation laws to generate the initial world state from a seed. The world is evaluated against playability gates (no runaway heat, traversable zones exist, fault activity is non-catastrophic). If gates fail, the server applies repair passes or regenerates with a derived seed.

---

## Player Causality Rule

> **The player cannot directly trigger GeoEvents, Climate changes, SDF modifications, or MaterialOperator transitions.**

The player's presence may:
- Cause the shell to absorb or lose material (via ShellOccupancy, driven by CharacterForces and surface contact).
- Leave shallow SDF depressions (footprints in loose material, cosmetic scale only).
- Locally displace aerosol dust (visual only).

The player's presence must not:
- Cause landslides by proximity.
- Alter wind or temperature fields.
- Accelerate or trigger any geological event.
- Trigger fault rupture.

All large-scale events are consequences of the simulation fields, not the player.

---

## Boundary Definitions

| System | Inputs | Outputs | Cannot receive from |
|--------|--------|---------|---------------------|
| AstroSystem | time | sun/moon angles, eclipse, tidal, impact flux | player |
| PlanetTileGrid | AstroSystem, PlateMap, FaultGraph | global fields | player |
| PlateMap | time, seed | plate assignments, boundary types | player |
| FaultGraph | tidal stress, thermal stress, plate motion | rupture events, signal flow | player |
| InsolationField | AstroSystem, RingShadow, PlanetSpin | energy field | player, GeoEvents |
| Climate | InsolationField | temp/pressure/wind/humidity/dust | player, SDFEdits |
| MaterialOperators | Climate, MaterialState | updated MaterialState | player |
| GeoRiskFields | Climate, SDFEdits, FaultGraph | risk fields | player |
| GeoEvents | GeoRiskFields | event spawns | player |
| SDFEdits | GeoEvents, RuptureEvent, ImpactEvent | terrain patches | player (except cosmetic footprints) |
| CharacterForces | Climate, SDFEdits, MaterialState | force vectors | player |
| CharacterController | CharacterForces, PlayerIntent | position/state | direct event triggers |
| ShellOccupancy | CharacterController, MaterialOperators | slot fill/loss/regen | player (player has no direct shell control) |
