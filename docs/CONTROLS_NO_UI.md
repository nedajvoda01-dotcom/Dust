# BARYCENTER — Controls & No-UI Contract

This document defines what input the player is permitted to provide and how the character responds to that input. Everything else is autonomous.

---

## Permitted Inputs

| Input | Description |
|-------|-------------|
| Move direction | Analogue or digital directional intent (forward / back / strafe). Magnitude matters. |
| Turn / look | Camera and character yaw. Continuous, not snapped. |
| Camera pitch | Vertical look angle. Cannot exceed gimbal limits. |
| System pause | Optional: pause simulation clock for accessibility. No in-world indication shown. |

**That is the complete list.** There are no other inputs.

---

## Forbidden Inputs

The following must not exist as player inputs under any circumstances:

- [ ] Interact / use / examine button
- [ ] Jump / climb button (climbing is a reflex state, not a button)
- [ ] Sprint toggle / stamina mechanic
- [ ] Inventory open / item use
- [ ] Map open
- [ ] Waypoint / marker placement
- [ ] Any form of "action" button
- [ ] Any menu that appears in the world view (pause menu must be outside simulation frame)

---

## Intent Model

The player provides **intent**: a desired direction and speed of movement. The character converts this intent into actual motion through physics negotiation:

```
PlayerIntent (direction, magnitude)
    + CharacterForces (wind, slope, friction, quakes)
    + CharacterState (balance, reflex, surface contact)
    ──────────────────────────────────────────────────
    → ResultantVelocity → NewPosition
```

The character is not a cursor. Issuing "move forward" does not guarantee forward movement.

---

## Autonomous Character States

These states activate without player input when physical conditions are met:

| State | Trigger condition | Player experience |
|-------|------------------|-------------------|
| `stumble` | Sudden terrain discontinuity (step, loose rock) | Brief loss of direction control |
| `slide` | Surface friction below threshold + slope | Character slides in slope direction, intent partially overridden |
| `crouch_reflex` | Incoming high-force wind gust | Character lowers centre of mass; move speed reduced |
| `grab` | Handhold geometry within reach during slide/fall near surface | Character arrests fall automatically |
| `fall_controlled` | No surface contact, velocity > threshold | Character orients feet-down, arms out; some aerial steering allowed |
| `recover` | After stumble/slide/fall, conditions allow | Character regains upright stance; intent resumes normally |
| `brace` | Quake impulse exceeds threshold | Character widens stance; brief input resistance |

The player cannot cancel these states by issuing intent. Intent resumes full effect when the state resolves.

---

## Environmental Precursor Signals (No UI)

Danger must be communicated through the world itself, never through screen overlays. Required precursors:

| Incoming event | Precursor signals |
|---------------|-------------------|
| Wind gust / storm | Dust begins moving on the ground, distant haze thickens, audio low-frequency tone rises |
| Landslide | Small rocks begin rolling uphill of the slope, subtle ground vibration (controller haptic if available), distant crack sound |
| Rift / fault event | Ground surface develops fine cracks (SDF surface detail change), low rumble, slight screen shake driven by physics (not scripted) |
| Void collapse | Hollow resonance when character is above void area, subtle ground sag, audio pitch of footsteps changes |
| Eclipse | Light temperature shift becomes visible, primary sun disk edge visibly occluded |
| Extreme cold | Character movement becomes stiffer (reduced intent-to-velocity conversion), breath condensation particle effect on exhale |

---

## UI Prohibition

Zero persistent screen-space UI is permitted in the simulation view. This includes:

- [ ] Health / stamina bars
- [ ] Compass or bearing indicator
- [ ] Altitude / speed readout
- [ ] Crosshair or aim indicator
- [ ] Item pickup prompts
- [ ] Any floating world-space text or icon
- [ ] Subtitles or narrative text

The simulation view is the entire screen. If a system pause screen is implemented, it must not appear as an overlay on the simulation frame — it must replace it.
