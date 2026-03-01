# BARYCENTER — Simulation Contract

This document defines the closed causal loop that governs all dynamic behaviour in BARYCENTER. Every runtime event must be traceable to this graph. Anything not in this graph does not happen.

---

## Causal Graph

```
AstroSystem
  └─► PlanetSpin
  └─► BinarySunAngles
  └─► EclipseState
  └─► RingShadow

BinarySunAngles + EclipseState
  └─► Insolation (surface energy field)

RingShadow
  └─► Insolation (shadow bands modulate field)

PlanetSpin
  └─► Insolation (day/night cycle)

Insolation
  └─► Climate
        ├─► Temperature field
        ├─► Pressure field
        ├─► Wind field (magnitude + direction)
        ├─► Humidity field
        └─► Dust suspension field

Climate
  └─► GeoRiskFields
        ├─► stress         (thermal expansion / contraction cycles)
        ├─► slope_instability (erosion + freeze/thaw + dust loading)
        └─► void_risk      (subsurface pressure / dissolution)

GeoRiskFields (when thresholds exceeded)
  └─► GeoEvents
        ├─► RiftEvent      (rare: fault opening, terrain split)
        ├─► LandslideEvent (moderate: slope failure, debris flow)
        ├─► CollapseEvent  (rare: void collapse, sinkhole)
        └─► ScreeEvent     (frequent: small rolling debris)

GeoEvents
  └─► SDFEdits (permanent deformation of terrain SDF field)
        ├─► surface height modifications
        └─► cave/void modifications

Climate + SDFEdits
  └─► CharacterForces
        ├─► wind_push      (from Wind field, direction + magnitude)
        ├─► snow_drag      (from Temperature + Humidity fields)
        ├─► friction       (from surface material: slope/dust/ice/wet)
        ├─► slope_gravity  (from terrain normal under character)
        └─► quake_impulse  (from RiftEvent / LandslideEvent)

CharacterForces + PlayerIntent
  └─► CharacterController
        ├─► balance_state  (upright / leaning / stumbling / fallen)
        ├─► reflex_state   (grabbing / crouching / sliding / recovering)
        ├─► velocity       (resultant of intent + forces)
        └─► position       (integrated from velocity)
```

---

## System Descriptions

### AstroSystem
Computes positions of both suns and the moon relative to the planet at each simulation tick. Outputs:
- `sun_primary_angle` (azimuth, elevation)
- `sun_secondary_angle` (azimuth, elevation)
- `eclipse_fraction` (0–1, how much primary is occluded by secondary or moon)
- `moon_position` (world-space vector)

### PlanetSpin
Integrates planet rotation over time. Controls local solar angle (day/night). Period defined by `day.length_minutes` in config.

### RingShadow
Projects ring geometry onto the planet surface as a shadow mask field. Band width and position are determined by ring geometry parameters and sun angle. Output is a 2D field of shadow intensity on the surface.

### Insolation
Combines sun angles, eclipse state, ring shadow, and day/night into a surface energy field (W/m² equivalent in simulation units). This field is the single source of thermal truth for the Climate system.

### Climate
Derives temperature, pressure, wind, humidity, and dust fields from Insolation. These fields evolve over time with inertia (thermal mass, momentum). They do not change instantly when Insolation changes.

### GeoRiskFields
Three scalar fields computed from Climate history and SDF terrain data:
- **stress**: accumulated thermal cycling load
- **slope_instability**: erosion + overburden + saturation
- **void_risk**: subsurface pressure differentials

### GeoEvents
When any GeoRiskField exceeds its threshold (defined in config), a GeoEvent is spawned at that location. Events are not scripted. Thresholds may be randomised within config ranges using the world seed.

### SDFEdits
Permanent modifications to the terrain SDF field. Stored as delta patches (not full-field copies). Applied when GeoEvents resolve. Caves and voids are part of the SDF.

### CharacterForces
A set of per-frame force vectors accumulated from environment systems and applied to the character physics body. The player does not control these forces. They are consequences of location and time.

### CharacterController
Receives `PlayerIntent` (direction + magnitude of movement desire, camera direction) and `CharacterForces` (from environment). Computes final movement, with reflex states overriding intent when forces exceed balance thresholds.

---

## Player Causality Rule

> **The player cannot directly trigger GeoEvents, Climate changes, or SDF modifications.**

The player's presence may:
- Leave footprints in dust (shallow SDF depression, purely cosmetic scale).
- Locally displace dust particles (visual only).

The player's presence must not:
- Cause landslides by proximity.
- Alter wind or temperature fields.
- Accelerate or trigger any geological event.

All large-scale events are consequences of the simulation fields, not the player.

---

## Boundary Definitions

| System | Inputs | Outputs | Cannot receive from |
|--------|--------|---------|---------------------|
| AstroSystem | time | sun/moon angles, eclipse | player |
| Insolation | AstroSystem, RingShadow, PlanetSpin | energy field | player, GeoEvents |
| Climate | Insolation | temp/pressure/wind/humidity/dust | player, SDFEdits |
| GeoRiskFields | Climate, SDFEdits | risk fields | player |
| GeoEvents | GeoRiskFields | event spawns | player |
| SDFEdits | GeoEvents | terrain patches | player (except cosmetic footprints) |
| CharacterForces | Climate, SDFEdits | force vectors | player |
| CharacterController | CharacterForces, PlayerIntent | position/state | direct event triggers |
