# DUST — Style Guide

This document defines hard visual rules. Any rendering decision that conflicts with these rules is invalid, regardless of technical convenience.

---

## 1. Frame Composition

### Three depth layers (mandatory)
Every frame must read as three distinct depth zones:

| Layer | Description | Treatment |
|-------|-------------|-----------|
| Foreground | Dark, high-contrast silhouettes and surfaces | Desaturated, cool-shifted, strong shadow |
| Midground | Soft mid-tones, readable terrain forms | Neutral temperature, moderate detail |
| Background | Pale, pastel, atmosphere-dissolved shapes | Warm haze, low contrast, high value |

Frames that collapse these three zones into two or one are invalid.

### Distant anchor (mandatory)
Every composition must contain at least one far silhouette that anchors the horizon: a monolith, a mesa, an eroded titan, a distant ridge. The anchor establishes scale and keeps the horizon from feeling empty in an unintended way.

### Large forms over small noise
Macro shapes (crater rims, dune fields, ridge lines) take visual priority over fine surface noise. Surface micro-detail must not compete with or fragment the large-form read at any resolution.

### Deliberate emptiness
Sky and ground negative space are compositional assets, not areas to fill. At least one third of the frame should be unoccupied sky or flat terrain. Never distribute detail uniformly across the frame.

---

## 2. Lighting

### Sun angle
The primary sun must sit low on the horizon (below 35° elevation) during the main "active" period. High-noon lighting is allowed only as a brief mid-day transition, never the default composition state.

### Warm key, cold fill
- Key light (primary sun): warm — amber to deep orange range.
- Fill / shadow: cool — blue-grey to violet-grey range.
- The temperature contrast is the primary tool for depth and form.

### Atmospheric perspective (haze)
Strong atmospheric scattering is required. Distant objects must be visibly lighter, warmer, and less saturated than near objects. Haze density must be strong enough that the background layer reads as a distinct tonal zone from the midground.

### Cinematic exposure
- Highlights must be soft — no hard, blown-out white areas except on specular surfaces of specific geometry.
- Exposure must not fluctuate frame-to-frame except during eclipse transitions or storm onset.
- Tone mapping must favor the shadow/midtone range, not the highlight range.

### Secondary sun
The secondary sun contributes a distinct cool or differently-angled light source. Its light may produce colored rim light on terrain facing away from the primary source. During eclipse or occlusion events, the balance of warm/cool shifts dramatically — this is a designed moment.

---

## 3. Pixelation

### Post-calculation pixelation
Pixel rendering is applied **after** full lighting, atmosphere, and post-process calculations. The pixelation stage does not replace shading — it quantizes the already-computed image.

### Downscale/upscale pipeline
The frame is rendered at reduced internal resolution and upscaled with nearest-neighbor interpolation. The downscale factor must be tunable (see `CONFIG_DEFAULTS.json`). Fractional factors are not permitted.

### Dithering
Ordered (Bayer matrix) dithering is applied to smooth gradients into structured patterns at the pixel scale. Dithering strength must be tunable. Random noise dithering is not permitted — structure is required.

### Color depth / palette
Color depth is limited in the pixelation stage. The palette is not fixed to a specific count, but the effective range must produce visible banding and quantization. The target feel is "limited palette film still," not "8-bit game."

### Light banding
3–5 discrete light bands on surfaces are permitted and encouraged where they reinforce form without destroying atmospheric softness. Banding must not appear on sky gradients. If banding destroys atmosphere-layer readability, it must be suppressed.

---

## 4. Prohibited Visual Patterns

- [ ] Uniform detail distribution across the whole frame.
- [ ] High-noon dominant lighting as a default state.
- [ ] Hard white specular highlights larger than a few pixels.
- [ ] Flickering or temporally unstable exposure.
- [ ] Noise that breaks up large-form silhouettes.
- [ ] Any bitmap texture used as albedo, normal, roughness, or mask.
- [ ] UI overlays, icons, or any screen-space non-diegetic element.
- [ ] Frame-space vignette that feels "applied" rather than physically motivated.
- [ ] Floating point / HDR bloom that washes out the color palette.
