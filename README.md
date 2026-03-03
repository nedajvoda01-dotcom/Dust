# DUST

**Source:** <https://github.com/nedajvoda01-dotcom/Dust>

---

DUST is a simulator of a living planet with no interface and no objectives. Two stars, a ring, and a moon shape the climate, snow, and tidal stress; plates and faults tear the crust; lava flows and crusts over; dust lifts and settles. You are a core that assembles a body from the material of the world: sand crumbles away, snow blows off, lava runs down you. There is no death — only loss of form and regeneration wherever physics allows. Two players can wander this world in silence, feeling it through sound, camera, and movement.

**Architecture:** server-authoritative 3D planetary field simulation. The server owns all world state and runs the full simulation at a fixed tick rate. Clients are thin: they send movement intent and receive world state snapshots for rendering.

---

## Run locally

```bash
pip install -r requirements.txt
python server.py          # starts on http://localhost:8765
```

Then open <http://localhost:8765> in a browser.

---

## Documentation

| Document | Contents |
|----------|----------|
| [MANIFEST](docs/MANIFEST.md) | Project philosophy and experience contract |
| [SIMULATION_CONTRACT](docs/SIMULATION_CONTRACT.md) | Closed causal graph — all dynamic behaviour |
| [ROADMAP](docs/ROADMAP.md) | Development stages |
| [TECH_CONSTRAINTS](docs/TECH_CONSTRAINTS.md) | Hard technical rules |
| [CONTROLS_NO_UI](docs/CONTROLS_NO_UI.md) | Input model and UI prohibition |
| [STYLE_GUIDE](docs/STYLE_GUIDE.md) | Visual rules |
| [VALIDATION_RULES](docs/VALIDATION_RULES.md) | Checkable rules for milestones |