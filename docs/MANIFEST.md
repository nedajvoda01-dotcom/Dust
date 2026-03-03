# DUST — Project Manifest

**Working title:** DUST  
**One-sentence description:** A server-authoritative, deterministic, fully physical simulator of an aggressive planet, where players with no interface control a core creature that temporarily assembles a body from the material underfoot, while the world lives on its own — geology, climate, snow, lava, faults, and celestial cycles without scripted events.

**What the player does:**
- **Walks** — moves through a living planet shaped by wind, geology, orbital mechanics, and the material that constitutes the character's own body.
- **Reacts** — the character's body responds to slope, gusts, quakes, and collapsing ground without player command; the shell gains and loses mass according to the same physics rules as the terrain.
- **Observes** — watches the sky, the light, the silhouettes; reads the world through sound, camera, and the feel of movement, not through interface.

---

## Philosophy

### The world is the subject; the player is a guest

DUST is not a game about accomplishment. There are no missions, no objectives, no map, no inventory, no markers. The planet does not acknowledge the player's presence. It orbits, breathes, erodes, and collapses according to its own internal rules. The player enters this process and endures it.

### Radical absence of guidance

No UI overlays. No waypoints. No tutorial text. No minimap. The only signals are environmental: light, sound, vibration, wind direction, dust density, and the position of two suns on the horizon. If the player does not know what to do, the answer is to watch and wait.

### Solitude and aggressive environment

The world is not hostile by design decision — it is hostile by physics. Insolation drives climate. Climate drives storms and pressure differentials. Pressure drives geological stress. Stress drives collapse and fault events. None of these are scripted. The player experiences them as a natural consequence of being present at a particular location at a particular orbital moment.

### Events are fields, not triggers

Nothing is scripted. A landslide happens because cumulative slope instability crossed a threshold. A dust storm arrives because thermal gradients exceeded wind thresholds. A shadow band crosses the terrain because the ring's geometry intersects the angle of the primary sun. All causality is physical.

### One law for everything

The material on the ground and the material that forms the player's shell are the same material, subject to the same operators. There are no game-specific exceptions: if lava melts rock on the terrain, it melts the shell too. If wind lifts loose regolith, it strips the shell too. The player is made of the world.

### Incomplete obedience

The player provides **intent**: a desired direction and speed of movement. The character interprets this intent through physics. A steep slope slows progress. A gust deflects trajectory. Ice removes friction. A tremor interrupts balance. The character may stumble, slide, crouch reflexively, or grab a ledge without player instruction. Intent is input; outcome is negotiated.

### Server is authoritative

The world is always simulated on the server. Clients are thin: they send intent and receive state. There is no client-side prediction of simulation outcomes — the server owns all causality.

---

## Player Entity: Core + Virtual Skeleton + Shell

### CoreSeed

The player is a core — a small, unchanging identity seed. The core has no death state. When form is lost, the core persists and regenerates a new shell where physics allows.

### Virtual Skeleton

The skeleton is a constraint graph: slots, frames, constraints, actuators, and effectors. "Limbs" are structures of slots and constraints. No pre-authored skeleton mesh exists; the skeleton is an abstract physical graph.

### Shell

The shell fills the volume envelopes defined by skeleton slots. The shell is material taken from the world and is subject to all the same physical operators as the terrain: it can be melted, frozen, blown away, or crushed. The dominant material of the shell is determined by what the core is standing on.

---

## Experience Contract

The player should feel:
- Small inside a vast, indifferent, geometrically coherent world.
- That time is real: the position of suns changes measurably, shadows move, temperature shifts.
- That danger is readable (sound, dust, vibration) but never labeled.
- That the planet was here before them and will be here after them.
- That their body is part of the world, not separate from it.

The player should never feel:
- Guided, prompted, or rewarded with UI feedback.
- That the world was constructed around them.
- That events were placed or scripted for dramatic effect.
- That their body is exempt from the rules that govern the terrain.
