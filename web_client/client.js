/**
 * client.js — Dust 3D Field Core browser client.
 *
 * Boot stages (no UI, no menu):
 *   LoadMinimal    — init renderer (WebGL2), spawn worker
 *   WorldHandshake — receive WORLD_SYNC from server, check buildId
 *   StartPlayable  — receive WORLD_BASELINE, enable input
 *   Running        — full 3D render + intent loop
 *
 * Rendering:
 *   WebGL2 3D scene: sphere planet (UV mesh) + SDF patch deformation +
 *   remote player blobs. No HUD or UI text in frame — use console.log.
 *
 * Network:
 *   Managed in worker.js. Main thread sends INTENT, receives world events.
 *
 * New message types (3D Field Core):
 *   WORLD_BASELINE   — planet radius, revisions, all patches, body graph
 *   SDF_PATCH_BATCH  — incremental patch deltas
 *   FIELDS_SNAPSHOT  — field generator parameters
 *   PLAYER_SNAPSHOT  — authoritative player positions from World3D
 *   REJOIN_RESYNC    — extended with sdfPatches
 */
"use strict";

// ──────────────────────────────────────────────────────────────────────────
// Build ID — replaced by build_web.sh during bundling
// ──────────────────────────────────────────────────────────────────────────
const CLIENT_BUILD_ID = "__BUILD_ID__";

// ──────────────────────────────────────────────────────────────────────────
// Boot stages
// ──────────────────────────────────────────────────────────────────────────
const Stage = Object.freeze({
  LoadMinimal:    "LoadMinimal",
  WorldHandshake: "WorldHandshake",
  StartPlayable:  "StartPlayable",
  StreamRest:     "StreamRest",
  Running:        "Running",
});

let _stage = Stage.LoadMinimal;

// ──────────────────────────────────────────────────────────────────────────
// Config / state
// ──────────────────────────────────────────────────────────────────────────
const INTERP_DELAY_MS        = 200;
const MAX_EXTRAP_MS          = 150;
const HARD_RESYNC_THRESHOLD  = 5.0;
const AUTO_RELOAD_THROTTLE   = 10_000;  // ms — prevent reload loops

let _lastReloadAttempt = 0;

const world = {
  seed:           null,
  worldId:        null,
  buildId:        null,
  simTime:        0,
  timeScale:      1,
  epoch:          0,
  spawnPos:       null,
  anchor:         null,
  storms:         [],
  globalDust:     0,
  // 3D Field Core
  planetRadius:   1000.0,
  sdfRevision:    0,
  fieldsRevision: 0,
  sdfPatches:     [],     // [{patch_id, cx, cy, cz, radius, strength, kind}]
  fieldsSnapshot: null,   // generator params for the client-side field estimator
};

const player = {
  id:    null,
  pos:   [0, 1001.8, 0],
  vel:   [0, 0, 0],
  flags: 0,
};

const remotePlayers = new Map();

// PLL time sync
let _pllTimeOffset     = 0;
let _pllRateCorrection = 0;
const PLL_KP             = 0.05;
const PLL_KI             = 0.001;
const PLL_INTEGRAL_CLAMP = 0.05;

let _lastPatchRevision = -1;
let _lastSeenEpoch = 0;
let _lastEventId   = -1;

// ──────────────────────────────────────────────────────────────────────────
// Auto-quality (Stage 14 / 23)
// ──────────────────────────────────────────────────────────────────────────
const perf = {
  autoQuality:     true,
  downscaleFactor: 1,
  fpsHistory:      [],
  lastFrameTime:   performance.now(),
  frameCount:      0,
};
const FPS_SAMPLE = 60;
const FPS_LOW    = 28;

function _autoQualityTick(now) {
  perf.fpsHistory.push(now - perf.lastFrameTime);
  perf.lastFrameTime = now;
  if (perf.fpsHistory.length > FPS_SAMPLE) perf.fpsHistory.shift();
  if (perf.fpsHistory.length < FPS_SAMPLE) return;
  const avg = perf.fpsHistory.reduce((a, b) => a + b, 0) / perf.fpsHistory.length;
  const fps = 1000 / avg;
  if (fps < FPS_LOW) {
    perf.downscaleFactor = Math.min(perf.downscaleFactor + 1, 4);
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Canvas + renderer detection
// ──────────────────────────────────────────────────────────────────────────
const canvas = document.getElementById("canvas");
let _gpuDevice   = null;
let _glCtx       = null;
let _renderMode  = "none";

async function _initRenderer() {
  // WebGL2 — 3D Field Core pipeline
  _glCtx = canvas.getContext("webgl2");
  if (_glCtx) {
    _renderMode = "webgl2";
    console.info("[Dust] Renderer: WebGL2 (3D Field Core)");
    _initWebGL3D(_glCtx);
    return true;
  }
  // Legacy WebGPU path removed in this pass — fall through to error.
  console.error("[Dust] No WebGL2 available — cannot render.");
  return false;
}

function _resize() {
  const dpr    = window.devicePixelRatio || 1;
  const factor = perf.downscaleFactor;
  canvas.width  = Math.floor(window.innerWidth  * dpr / factor);
  canvas.height = Math.floor(window.innerHeight * dpr / factor);
  canvas.style.width  = window.innerWidth  + "px";
  canvas.style.height = window.innerHeight + "px";
}
window.addEventListener("resize", _resize);

// ──────────────────────────────────────────────────────────────────────────
// Input
// ──────────────────────────────────────────────────────────────────────────
const keys = new Set();
let _pointerLocked = false;
let _mouseDX = 0, _mouseDY = 0;

window.addEventListener("keydown", (e) => {
  keys.add(e.code);
  // Lock pointer on first user gesture after StartPlayable
  if (_stage !== Stage.LoadMinimal && _stage !== Stage.WorldHandshake) {
    if (!_pointerLocked) canvas.requestPointerLock();
  }
});
window.addEventListener("keyup",   (e) => keys.delete(e.code));
document.addEventListener("pointerlockchange", () => {
  _pointerLocked = document.pointerLockElement === canvas;
});
window.addEventListener("mousemove", (e) => {
  if (_pointerLocked) { _mouseDX += e.movementX; _mouseDY += e.movementY; }
});

// ──────────────────────────────────────────────────────────────────────────
// Network worker
// ──────────────────────────────────────────────────────────────────────────
let _worker = null;
let _sendQ  = [];

function _initWorker() {
  _worker = new Worker("worker.js");
  _worker.onmessage = _onWorkerMessage;
  const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl   = `${wsProto}//${location.host}/ws`;
  _worker.postMessage({ type: "CONNECT", url: wsUrl });
}

function _send(payload) {
  if (_worker) {
    _worker.postMessage({ type: "SEND", payload });
  }
}

// ──────────────────────────────────────────────────────────────────────────
// Build-ID mismatch reload (with throttle)
// ──────────────────────────────────────────────────────────────────────────
function _checkBuildId(serverBuildId) {
  if (!CLIENT_BUILD_ID || CLIENT_BUILD_ID === "__BUILD_ID__") return;
  if (!serverBuildId)                                          return;
  if (serverBuildId === CLIENT_BUILD_ID)                       return;
  const now = Date.now();
  if (now - _lastReloadAttempt < AUTO_RELOAD_THROTTLE) return;
  _lastReloadAttempt = now;
  console.warn("[Dust] Build mismatch — reloading…", { CLIENT_BUILD_ID, serverBuildId });
  location.reload();
}

// ──────────────────────────────────────────────────────────────────────────
// Worker message handler (server events)
// ──────────────────────────────────────────────────────────────────────────
function _onWorkerMessage(ev) {
  const { type, data, code, reason } = ev.data;

  switch (type) {
    case "WS_OPEN":
      console.info("[Dust] WebSocket open");
      _send({ type: "JOIN" });
      _stage = Stage.WorldHandshake;
      break;

    case "WS_CLOSE":
      console.warn("[Dust] WebSocket closed", code, reason);
      _stage = Stage.WorldHandshake;
      break;

    case "WS_ERROR":
      console.error("[Dust] WebSocket error:", ev.data.message);
      break;

    case "WS_MSG":
      _onServerMsg(data);
      break;

    default:
      break;
  }
}

function _onServerMsg(msg) {
  if (!msg || !msg.type) return;

  switch (msg.type) {
    case "WORLD_SYNC": {
      world.seed      = msg.seed;
      world.worldId   = msg.worldId;
      world.buildId   = msg.buildId;
      world.simTime   = msg.simTime   ?? 0;
      world.timeScale = msg.timeScale ?? 1;
      world.spawnPos  = msg.spawnPos;
      world.anchor    = msg.anchor;
      if (msg.spawnPos) player.pos = msg.spawnPos.slice();
      // Apply any geo events already on the server
      if (Array.isArray(msg.geoEvents)) {
        msg.geoEvents.forEach(_applyGeoEvent);
      }
      _checkBuildId(msg.buildId);
      // WORLD_BASELINE follows immediately — stay in WorldHandshake
      console.info("[Dust] WORLD_SYNC received — awaiting WORLD_BASELINE");
      break;
    }

    case "WORLD_BASELINE": {
      // 3D Field Core — apply baseline planet + patches + body
      world.planetRadius   = msg.planetRadius   ?? 1000.0;
      world.sdfRevision    = msg.sdfRevision    ?? 0;
      world.fieldsRevision = msg.fieldsRevision ?? 0;
      world.fieldsSnapshot = msg.fieldsSnapshot ?? null;
      if (Array.isArray(msg.sdfPatches)) {
        world.sdfPatches = msg.sdfPatches.slice();
        _rebuildPlanetMesh();
      }
      if (msg.spawnPos && Array.isArray(msg.spawnPos)) {
        player.pos = msg.spawnPos.slice();
      } else {
        // Default spawn at north pole surface + hover
        player.pos = [0, world.planetRadius + 1.8, 0];
      }
      _lastPatchRevision = world.sdfRevision;
      _stage = Stage.StartPlayable;
      console.info(
        "[Dust] WORLD_BASELINE received — planet r=" + world.planetRadius +
        " patches=" + world.sdfPatches.length + " — stage StartPlayable"
      );
      break;
    }

    case "SDF_PATCH_BATCH": {
      // 3D Field Core — incremental SDF patches
      if (Array.isArray(msg.patches) && msg.patches.length > 0) {
        for (const p of msg.patches) {
          world.sdfPatches.push(p);
        }
        world.sdfRevision    = msg.sdfRevision ?? world.sdfRevision;
        _lastPatchRevision   = world.sdfRevision;
        _rebuildPlanetMesh();
        console.debug("[Dust] SDF_PATCH_BATCH:", msg.patches.length,
                      "new patches (rev=" + world.sdfRevision + ")");
      }
      break;
    }

    case "FIELDS_SNAPSHOT": {
      // 3D Field Core — update field generator params
      world.fieldsSnapshot  = msg.fieldsSnapshot ?? null;
      world.fieldsRevision  = (msg.fieldsSnapshot && msg.fieldsSnapshot.fields_revision)
                              ?? world.fieldsRevision;
      break;
    }

    case "PLAYER_SNAPSHOT": {
      // 3D Field Core — authoritative player states from World3D
      if (Array.isArray(msg.players)) {
        const now = Date.now();
        msg.players.forEach((p) => {
          if (p.id === player.id) {
            // Authoritative position correction — snap if far, blend if close
            if (p.pos && Array.isArray(p.pos)) {
              const dx = p.pos[0] - player.pos[0];
              const dy = p.pos[1] - player.pos[1];
              const dz = p.pos[2] - player.pos[2];
              const d2 = dx*dx + dy*dy + dz*dz;
              if (d2 > 25.0) {
                player.pos = p.pos.slice(); // snap
              }
            }
            return;
          }
          remotePlayers.set(p.id, { ...p, _ts: now });
        });
      }
      break;
    }

    case "WORLD_TICK": {
      const serverTime = msg.simTime + _pllTimeOffset;
      const err = serverTime - (world.simTime + _pllRateCorrection);
      if (Math.abs(err) > HARD_RESYNC_THRESHOLD) {
        world.simTime = msg.simTime;
        _pllTimeOffset = _pllRateCorrection = 0;
      } else {
        _pllRateCorrection += PLL_KI * err;
        _pllRateCorrection  = Math.max(-PLL_INTEGRAL_CLAMP,
          Math.min(PLL_INTEGRAL_CLAMP, _pllRateCorrection));
        _pllTimeOffset += PLL_KP * err;
        world.simTime  += _pllRateCorrection;
      }
      world.timeScale = msg.timeScale ?? 1;
      if (msg.epoch && msg.epoch > _lastSeenEpoch) {
        _lastSeenEpoch = msg.epoch;
        world.epoch    = msg.epoch;
      }
      if (_stage === Stage.StartPlayable || _stage === Stage.Running) {
        _stage = Stage.Running;
      }
      break;
    }

    case "PLAYERS":
      if (Array.isArray(msg.players)) {
        const now = Date.now();
        msg.players.forEach((p) => {
          if (p.id === player.id) return;
          remotePlayers.set(p.id, { ...p, _ts: now });
        });
      }
      break;

    case "GEO_EVENT":
      _applyGeoEvent(msg);
      break;

    case "CLIMATE_SNAP":
      world.storms     = msg.storms     ?? [];
      world.globalDust = msg.globalDust ?? 0;
      break;

    case "REJOIN_RESYNC":
      world.simTime = msg.simTime ?? world.simTime;
      world.epoch   = msg.epoch   ?? world.epoch;
      if (Array.isArray(msg.catchupEvents)) {
        msg.catchupEvents.forEach(_applyGeoEvent);
      }
      // 3D Field Core — apply catch-up SDF patches
      if (Array.isArray(msg.sdfPatches) && msg.sdfPatches.length > 0) {
        for (const p of msg.sdfPatches) world.sdfPatches.push(p);
        world.sdfRevision  = msg.sdfRevision ?? world.sdfRevision;
        _lastPatchRevision = world.sdfRevision;
        _rebuildPlanetMesh();
        console.info("[Dust] REJOIN_RESYNC: caught up", msg.sdfPatches.length, "SDF patches");
      }
      break;

    case "PING":
      // Server-initiated ping — reply with PONG including RTT measurement
      _send({ type: "PONG", t: msg.t });
      break;

    case "PONG":
      // Response to our own ping (RTT)
      if (msg.t != null) {
        const rtt = Date.now() / 1000 - msg.t;
        console.debug("[Dust] RTT:", (rtt * 1000).toFixed(1), "ms");
      }
      break;

    default:
      break;
  }
}

function _applyGeoEvent(ev) {
  if (!ev) return;
  const id = parseInt(ev.eventId ?? -1, 10);
  if (id > _lastEventId) _lastEventId = id;
}

// ──────────────────────────────────────────────────────────────────────────
// Render loop
// ──────────────────────────────────────────────────────────────────────────
let _lastFrameTS = 0;

function _renderFrame(ts) {
  requestAnimationFrame(_renderFrame);
  const dt = Math.min((ts - _lastFrameTS) / 1000, 0.1);
  _lastFrameTS = ts;

  if (perf.autoQuality) _autoQualityTick(ts);

  // Advance local player simulation and send intent
  if (_stage === Stage.StartPlayable || _stage === Stage.Running) {
    _tickInput(dt);
    _sendIntent();
  }

  _draw();
}

// ──────────────────────────────────────────────────────────────────────────
// Simulation tick helpers
// ──────────────────────────────────────────────────────────────────────────
const CAM = { yaw: 0, pitch: 0 };
const MOUSE_SENS = 0.002;
const MOVE_SPEED = 5.0;
let   _moveVec   = [0, 0];   // [x, z] intent from WASD

function _tickInput(dt) {
  // Mouse look
  CAM.yaw   -= _mouseDX * MOUSE_SENS;
  CAM.pitch  = Math.max(-1.4, Math.min(1.4, CAM.pitch - _mouseDY * MOUSE_SENS));
  _mouseDX   = 0;
  _mouseDY   = 0;

  // Movement (WASD)
  let mx = 0, mz = 0;
  if (keys.has("KeyW")) mz += 1;
  if (keys.has("KeyS")) mz -= 1;
  if (keys.has("KeyA")) mx -= 1;
  if (keys.has("KeyD")) mx += 1;
  const mlen = Math.sqrt(mx*mx + mz*mz) || 1;
  _moveVec = [mx / mlen, mz / mlen];

  // Local position prediction (corrected by PLAYER_SNAPSHOT from server)
  const fwd_x = Math.sin(CAM.yaw);
  const fwd_z = Math.cos(CAM.yaw);
  const rgt_x = Math.cos(CAM.yaw);
  const rgt_z = -Math.sin(CAM.yaw);
  const dx = (fwd_x * _moveVec[1] + rgt_x * _moveVec[0]) * MOVE_SPEED;
  const dz = (fwd_z * _moveVec[1] + rgt_z * _moveVec[0]) * MOVE_SPEED;

  player.vel = [dx, 0, dz];
  player.pos[0] += dx * dt;
  player.pos[2] += dz * dt;

  // Keep on sphere surface
  const r = world.planetRadius + 1.8;
  const pr = Math.sqrt(
    player.pos[0]*player.pos[0] +
    player.pos[1]*player.pos[1] +
    player.pos[2]*player.pos[2]
  ) || 1;
  const scale = r / pr;
  player.pos[0] *= scale;
  player.pos[1] *= scale;
  player.pos[2] *= scale;
}

let _lastSendTS = 0;
const SEND_INTERVAL = 1000 / 20; // 20 Hz

function _sendIntent() {
  const now = performance.now();
  if (now - _lastSendTS < SEND_INTERVAL) return;
  _lastSendTS = now;
  _send({
    type:       "INTENT",
    moveVec:    _moveVec,
    lookYaw:    CAM.yaw,
    lookPitch:  CAM.pitch,
    ctrl:       0,
    r:          keys.has("KeyR"),
    clientTime: now / 1000,
  });
}

// ──────────────────────────────────────────────────────────────────────────
// WebGL 3D Renderer
// ──────────────────────────────────────────────────────────────────────────

// Renderer state
let _gl3d        = null;   // WebGL2 context alias
let _prog3d      = null;   // main shader program
let _planetVAO   = null;
let _planetVBO   = null;
let _planetIBO   = null;
let _planetIdxCount = 0;
let _blobVAO     = null;
let _blobVBO     = null;
let _blobIBO     = null;
let _blobIdxCount = 0;
// Patch uniforms — up to N patches in shader
const MAX_PATCHES = 16;

// ── Minimal mat4 helpers (column-major Float32Array) ────────────────────
function mat4Identity() {
  return new Float32Array([
    1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
  ]);
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[j + k*4] * b[k + i*4];
      out[j + i*4] = s;
    }
  }
  return out;
}

function mat4Perspective(fovY, aspect, near, far) {
  const f = 1.0 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0]  =  f / aspect;
  out[5]  =  f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

function mat4LookAt(eye, centre, up) {
  const fx = centre[0]-eye[0], fy = centre[1]-eye[1], fz = centre[2]-eye[2];
  const fl = Math.sqrt(fx*fx+fy*fy+fz*fz)||1;
  const nx = fx/fl, ny = fy/fl, nz = fz/fl;
  const rx = ny*up[2]-nz*up[1], ry = nz*up[0]-nx*up[2], rz = nx*up[1]-ny*up[0];
  const rl = Math.sqrt(rx*rx+ry*ry+rz*rz)||1;
  const sx = rx/rl, sy = ry/rl, sz = rz/rl;
  const ux = sy*nz-sz*ny, uy = sz*nx-sx*nz, uz = sx*ny-sy*nx;
  const out = new Float32Array(16);
  out[0]=sx; out[1]=ux; out[2]=-nx; out[3]=0;
  out[4]=sy; out[5]=uy; out[6]=-ny; out[7]=0;
  out[8]=sz; out[9]=uz; out[10]=-nz; out[11]=0;
  out[12]=-(sx*eye[0]+sy*eye[1]+sz*eye[2]);
  out[13]=-(ux*eye[0]+uy*eye[1]+uz*eye[2]);
  out[14]=(nx*eye[0]+ny*eye[1]+nz*eye[2]);
  out[15]=1;
  return out;
}

function mat4Translate(tx, ty, tz) {
  const m = mat4Identity();
  m[12] = tx; m[13] = ty; m[14] = tz;
  return m;
}

function mat4Scale(s) {
  const m = mat4Identity();
  m[0] = m[5] = m[10] = s;
  return m;
}

// ── UV Sphere generation ────────────────────────────────────────────────
function _buildUVSphere(radius, stacks, slices) {
  const verts = [];
  const idx   = [];

  for (let i = 0; i <= stacks; i++) {
    const phi   = Math.PI * i / stacks;  // 0..π
    const sinP  = Math.sin(phi);
    const cosP  = Math.cos(phi);
    for (let j = 0; j <= slices; j++) {
      const theta = 2 * Math.PI * j / slices;
      const x = sinP * Math.cos(theta);
      const y = cosP;
      const z = sinP * Math.sin(theta);
      // pos (3), normal (3)
      verts.push(x*radius, y*radius, z*radius, x, y, z);
    }
  }

  for (let i = 0; i < stacks; i++) {
    for (let j = 0; j < slices; j++) {
      const a = i*(slices+1) + j;
      const b = a + (slices+1);
      idx.push(a, b, a+1,  b, b+1, a+1);
    }
  }

  return {
    vertices: new Float32Array(verts),
    indices:  new Uint16Array(idx),
  };
}

// ── Shader sources ───────────────────────────────────────────────────────
const _VERT_SRC = `#version 300 es
precision highp float;

in  vec3 aPos;
in  vec3 aNorm;

uniform mat4 uMVP;
uniform mat4 uModel;

// SDF patches: centre(xyz) + radius + strength
uniform vec4  uPatch[${MAX_PATCHES}];   // xyz=centre, w=radius
uniform float uPatchStr[${MAX_PATCHES}];
uniform int   uNumPatches;

out vec3 vNorm;
out vec3 vWorldPos;
out float vDisplace;

void main() {
  vec3 pos  = aPos;
  vec3 norm = aNorm;

  // Apply SDF patch displacements along the normal
  float totalDisplace = 0.0;
  for (int i = 0; i < uNumPatches; i++) {
    vec3  centre   = uPatch[i].xyz;
    float pRadius  = uPatch[i].w;
    float dist     = distance(pos, centre);
    if (dist < pRadius) {
      float t  = 1.0 - dist / pRadius;
      totalDisplace += uPatchStr[i] * t * t;
    }
  }
  pos      += norm * totalDisplace;
  vDisplace = totalDisplace;
  vNorm     = norm;
  vWorldPos = pos;
  gl_Position = uMVP * vec4(pos, 1.0);
}
`;

const _FRAG_SRC = `#version 300 es
precision mediump float;

in  vec3  vNorm;
in  vec3  vWorldPos;
in  float vDisplace;

uniform vec3  uLightDir;   // normalised world-space sun direction
uniform vec3  uBaseColor;
uniform float uDustDensity;

out vec4 fragColor;

void main() {
  vec3 norm   = normalize(vNorm);
  float ndotl = max(dot(norm, normalize(uLightDir)), 0.0);
  vec3 ambient = uBaseColor * 0.25;
  vec3 diffuse = uBaseColor * ndotl;

  // Dust tint
  vec3 dustColor = vec3(0.6, 0.50, 0.38);
  vec3 color = mix(diffuse + ambient, dustColor, uDustDensity * 0.4);

  // Patch highlight — show displacement as slight brightening
  color += vec3(vDisplace * 2.0);

  fragColor = vec4(color, 1.0);
}
`;

function _compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.error("[Dust] Shader compile error:", gl.getShaderInfoLog(sh));
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

function _buildProgram(gl, vertSrc, fragSrc) {
  const vs   = _compileShader(gl, gl.VERTEX_SHADER,   vertSrc);
  const fs   = _compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  if (!vs || !fs) return null;
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error("[Dust] Program link error:", gl.getProgramInfoLog(prog));
    return null;
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return prog;
}

function _uploadMesh(gl, vertices, indices) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);

  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);

  const ibo = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

  // aPos — 3 floats at offset 0
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
  // aNorm — 3 floats at offset 12
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);

  gl.bindVertexArray(null);
  return { vao, vbo, ibo };
}

function _initWebGL3D(gl) {
  _gl3d = gl;
  _prog3d = _buildProgram(gl, _VERT_SRC, _FRAG_SRC);
  if (!_prog3d) { console.error("[Dust] Failed to build shader program"); return; }

  // Bind attrib locations
  gl.bindAttribLocation(_prog3d, 0, "aPos");
  gl.bindAttribLocation(_prog3d, 1, "aNorm");

  // Planet mesh
  const planet = _buildUVSphere(world.planetRadius || 1000.0, 32, 64);
  const pm = _uploadMesh(gl, planet.vertices, planet.indices);
  _planetVAO = pm.vao; _planetVBO = pm.vbo; _planetIBO = pm.ibo;
  _planetIdxCount = planet.indices.length;

  // Blob mesh (small sphere for players)
  const blob = _buildUVSphere(1.5, 8, 16);
  const bm   = _uploadMesh(gl, blob.vertices, blob.indices);
  _blobVAO = bm.vao; _blobVBO = bm.vbo; _blobIBO = bm.ibo;
  _blobIdxCount = blob.indices.length;

  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.CULL_FACE);
  console.info("[Dust] WebGL3D initialized");
}

// Rebuild planet mesh when planet radius changes (on WORLD_BASELINE)
function _rebuildPlanetMesh() {
  if (!_gl3d || !_planetVBO) return;
  const gl = _gl3d;
  const r  = world.planetRadius || 1000.0;
  const planet = _buildUVSphere(r, 32, 64);
  gl.bindBuffer(gl.ARRAY_BUFFER, _planetVBO);
  gl.bufferData(gl.ARRAY_BUFFER, planet.vertices, gl.DYNAMIC_DRAW);
  _planetIdxCount = planet.indices.length;
}

// ── Draw ─────────────────────────────────────────────────────────────────
function _draw() {
  if (_renderMode !== "webgl2" || !_gl3d || !_prog3d) return;
  const gl = _gl3d;

  const W = canvas.width, H = canvas.height;
  gl.viewport(0, 0, W, H);

  // Sky colour: dark ochre tinted by dust
  const dustTint = world.globalDust;
  gl.clearColor(
    0.04 + dustTint * 0.08,
    0.03 + dustTint * 0.03,
    0.02,
    1.0
  );
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  if (_stage !== Stage.StartPlayable && _stage !== Stage.Running) return;

  // ── Camera ──────────────────────────────────────────────────────────
  const px = player.pos[0], py = player.pos[1], pz = player.pos[2];
  const pr = Math.sqrt(px*px+py*py+pz*pz) || 1;
  // Up = outward from planet centre
  const upX = px/pr, upY = py/pr, upZ = pz/pr;
  // Camera sits behind and above the player, relative to surface
  const backDist = 6.0, upDist = 3.0;
  const fwdX = Math.sin(CAM.yaw), fwdZ = Math.cos(CAM.yaw);
  const eyeX = px - fwdX*backDist + upX*upDist;
  const eyeY = py + upY*upDist;
  const eyeZ = pz - fwdZ*backDist + upZ*upDist;
  const tgtX = px + fwdX*4, tgtY = py, tgtZ = pz + fwdZ*4;

  const r  = world.planetRadius || 1000.0;
  const proj = mat4Perspective(
    Math.PI / 4,     // 45° fov
    W / H,
    r * 0.001,       // near
    r * 10           // far
  );
  const view  = mat4LookAt([eyeX, eyeY, eyeZ], [tgtX, tgtY, tgtZ],
                            [upX, upY, upZ]);
  const vp    = mat4Multiply(proj, view);

  // ── Patch uniforms ───────────────────────────────────────────────────
  const patches = world.sdfPatches.slice(-MAX_PATCHES);
  const nP      = patches.length;
  const pCentre = new Float32Array(MAX_PATCHES * 4);
  const pStr    = new Float32Array(MAX_PATCHES);
  for (let i = 0; i < nP; i++) {
    const p = patches[i];
    pCentre[i*4]   = p.cx     ?? 0;
    pCentre[i*4+1] = p.cy     ?? 0;
    pCentre[i*4+2] = p.cz     ?? 0;
    pCentre[i*4+3] = p.radius ?? 0.5;
    pStr[i]         = (p.kind === "sphere_deposit") ? -(p.strength ?? 0.08)
                                                     :  (p.strength ?? 0.08);
  }

  gl.useProgram(_prog3d);

  // Patch uniforms
  const uPatch    = gl.getUniformLocation(_prog3d, "uPatch");
  const uPatchStr = gl.getUniformLocation(_prog3d, "uPatchStr");
  const uNumP     = gl.getUniformLocation(_prog3d, "uNumPatches");
  if (uPatch)    gl.uniform4fv(uPatch,    pCentre);
  if (uPatchStr) gl.uniform1fv(uPatchStr, pStr);
  if (uNumP)     gl.uniform1i(uNumP,      nP);

  // Light: single sun at fixed azimuth + elevation
  const lx = 0.6, ly = 0.8, lz = 0.3;
  const ll = Math.sqrt(lx*lx+ly*ly+lz*lz);
  gl.uniform3f(gl.getUniformLocation(_prog3d, "uLightDir"), lx/ll, ly/ll, lz/ll);
  gl.uniform1f(gl.getUniformLocation(_prog3d, "uDustDensity"), world.globalDust);

  // ── Draw planet ──────────────────────────────────────────────────────
  const model   = mat4Identity();
  const mvp     = mat4Multiply(vp, model);
  gl.uniformMatrix4fv(gl.getUniformLocation(_prog3d, "uMVP"),   false, mvp);
  gl.uniformMatrix4fv(gl.getUniformLocation(_prog3d, "uModel"), false, model);
  gl.uniform3f(gl.getUniformLocation(_prog3d, "uBaseColor"), 0.35, 0.30, 0.24);

  gl.bindVertexArray(_planetVAO);
  gl.drawElements(gl.TRIANGLES, _planetIdxCount, gl.UNSIGNED_SHORT, 0);

  // ── Draw player blobs ────────────────────────────────────────────────
  // Disable patch displacement for blobs
  if (uNumP) gl.uniform1i(uNumP, 0);
  gl.uniform3f(gl.getUniformLocation(_prog3d, "uBaseColor"), 0.85, 0.75, 0.50);

  const drawBlob = (bx, by, bz) => {
    const bModel = mat4Multiply(mat4Translate(bx, by, bz), mat4Scale(1.0));
    const bMvp   = mat4Multiply(vp, bModel);
    gl.uniformMatrix4fv(gl.getUniformLocation(_prog3d, "uMVP"),   false, bMvp);
    gl.uniformMatrix4fv(gl.getUniformLocation(_prog3d, "uModel"), false, bModel);
    gl.bindVertexArray(_blobVAO);
    gl.drawElements(gl.TRIANGLES, _blobIdxCount, gl.UNSIGNED_SHORT, 0);
  };

  // Local player
  drawBlob(player.pos[0], player.pos[1], player.pos[2]);

  // Remote players
  const now = Date.now();
  for (const [pid, rp] of remotePlayers) {
    if (now - rp._ts > 5000) { remotePlayers.delete(pid); continue; }
    if (rp.pos && rp.pos.length === 3) drawBlob(rp.pos[0], rp.pos[1], rp.pos[2]);
  }

  gl.bindVertexArray(null);
}

// ──────────────────────────────────────────────────────────────────────────
// Service Worker registration
// ──────────────────────────────────────────────────────────────────────────
function _registerSW() {
  if (!("serviceWorker" in navigator)) return;
  navigator.serviceWorker.register("sw.js").then((reg) => {
    console.info("[Dust] Service Worker registered, scope:", reg.scope);
  }).catch((e) => {
    console.warn("[Dust] Service Worker registration failed:", e);
  });

  navigator.serviceWorker.addEventListener("message", (ev) => {
    if (ev.data && ev.data.type === "SW_UPDATED") {
      const now = Date.now();
      if (now - _lastReloadAttempt < AUTO_RELOAD_THROTTLE) return;
      _lastReloadAttempt = now;
      console.info("[Dust] SW updated — reloading…");
      location.reload();
    }
  });
}

// ──────────────────────────────────────────────────────────────────────────
// Boot
// ──────────────────────────────────────────────────────────────────────────
async function _boot() {
  console.info("[Dust] Boot — 3D Field Core");

  _resize();

  // Stage: LoadMinimal
  const ok = await _initRenderer();
  if (!ok) return;

  _registerSW();
  _initWorker();

  _lastFrameTS = performance.now();
  requestAnimationFrame(_renderFrame);

  // Periodic REJOIN_RESYNC ping (reconnects handled by worker)
  setInterval(() => {
    if (_stage === Stage.Running) {
      _send({
        type:                "REJOIN_RESYNC",
        lastSeenTick:        _lastSeenEpoch,
        lastEventId:         _lastEventId,
        lastPatchRevision:   _lastPatchRevision,
      });
    }
  }, 30_000);
}

_boot();
