/**
 * client.js — Dust Stage 23 browser client.
 *
 * Boot stages (no UI, no menu):
 *   LoadMinimal   — init renderer (WebGPU → WebGL2 fallback), spawn worker
 *   WorldHandshake — receive WORLD_SYNC from server, check buildId
 *   StartPlayable  — spawn player at baseline position, enable input
 *   StreamRest     — apply climate snapshot / catch-up events
 *
 * Rendering:
 *   Attempts WebGPU; falls back to WebGL2; logs error and stops on failure.
 *
 * Network:
 *   Managed entirely in worker.js (Web Worker).
 *   Main thread only sends input commands and receives world events.
 *
 * Cache / version:
 *   On WORLD_SYNC, compares server buildId to __BUILD_ID__ (injected
 *   by build_web.sh).  Mismatch triggers throttled location.reload().
 *
 * Service Worker:
 *   Registered on boot if supported.  SW_UPDATED → location.reload().
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
  seed:       null,
  worldId:    null,
  buildId:    null,
  simTime:    0,
  timeScale:  1,
  epoch:      0,
  spawnPos:   null,
  anchor:     null,
  storms:     [],
  globalDust: 0,
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
  // WebGPU
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        _gpuDevice  = await adapter.requestDevice();
        _renderMode = "webgpu";
        console.info("[Dust] Renderer: WebGPU");
        return true;
      }
    } catch (e) {
      console.warn("[Dust] WebGPU init failed:", e);
    }
  }
  // WebGL2 fallback
  _glCtx = canvas.getContext("webgl2");
  if (_glCtx) {
    _renderMode = "webgl2";
    console.info("[Dust] Renderer: WebGL2");
    return true;
  }
  console.error("[Dust] No WebGPU or WebGL2 available — cannot render.");
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
      // Advance to playable
      _stage = Stage.StartPlayable;
      console.info("[Dust] World synced — stage StartPlayable");
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
      if (_stage === Stage.StreamRest || _stage === Stage.Running) {
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
      if (_stage === Stage.StartPlayable) _stage = Stage.StreamRest;
      break;

    case "REJOIN_RESYNC":
      world.simTime = msg.simTime ?? world.simTime;
      world.epoch   = msg.epoch   ?? world.epoch;
      if (Array.isArray(msg.catchupEvents)) {
        msg.catchupEvents.forEach(_applyGeoEvent);
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

  // Advance local player simulation (simple placeholder)
  if (_stage === Stage.StartPlayable ||
      _stage === Stage.StreamRest    ||
      _stage === Stage.Running) {
    _tickInput(dt);
    _sendPlayerState();
  }

  _draw();
}

// ──────────────────────────────────────────────────────────────────────────
// Simulation tick helpers
// ──────────────────────────────────────────────────────────────────────────
const CAM = { yaw: 0, pitch: 0 };
const MOUSE_SENS = 0.002;
const MOVE_SPEED = 5.0;

function _tickInput(dt) {
  // Mouse look
  CAM.yaw   -= _mouseDX * MOUSE_SENS;
  CAM.pitch  = Math.max(-1.4, Math.min(1.4, CAM.pitch - _mouseDY * MOUSE_SENS));
  _mouseDX   = 0;
  _mouseDY   = 0;

  // Movement (WASD + Space/Shift)
  const fwd  = [Math.sin(CAM.yaw),  0, Math.cos(CAM.yaw)];
  const rgt  = [Math.cos(CAM.yaw),  0, -Math.sin(CAM.yaw)];
  let dx = 0, dy = 0, dz = 0;
  if (keys.has("KeyW")) { dx += fwd[0]; dz += fwd[2]; }
  if (keys.has("KeyS")) { dx -= fwd[0]; dz -= fwd[2]; }
  if (keys.has("KeyA")) { dx -= rgt[0]; dz -= rgt[2]; }
  if (keys.has("KeyD")) { dx += rgt[0]; dz += rgt[2]; }
  if (keys.has("Space"))      dy += 1;
  if (keys.has("ShiftLeft"))  dy -= 1;

  const len = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
  player.vel = [dx/len * MOVE_SPEED, dy/len * MOVE_SPEED, dz/len * MOVE_SPEED];
  player.pos[0] += player.vel[0] * dt;
  player.pos[1] += player.vel[1] * dt;
  player.pos[2] += player.vel[2] * dt;
}

let _lastSendTS = 0;
const SEND_INTERVAL = 1000 / 20; // 20 Hz

function _sendPlayerState() {
  const now = performance.now();
  if (now - _lastSendTS < SEND_INTERVAL) return;
  _lastSendTS = now;
  _send({
    type:  "PLAYER_STATE",
    pos:   player.pos,
    vel:   player.vel,
    flags: player.flags,
  });
}

// ──────────────────────────────────────────────────────────────────────────
// Draw
// ──────────────────────────────────────────────────────────────────────────
function _draw() {
  if (_renderMode === "webgl2" && _glCtx) {
    const gl = _glCtx;
    const r  = world.globalDust * 0.1;
    const g  = 0.03 + world.globalDust * 0.02;
    const b  = 0.02;
    gl.clearColor(r, g, b, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    // Full geometry rendering deferred to WASM pipeline
  } else if (_renderMode === "webgpu" && _gpuDevice) {
    // WebGPU frame submission handled by WASM pipeline
  }
}

// ──────────────────────────────────────────────────────────────────────────
// HUD (minimal — diagnostics only, no game UI per Stage constraints)
// ──────────────────────────────────────────────────────────────────────────
function _updateHUD() {
  const el = (id) => document.getElementById(id);
  const hWorld   = el("hud-world");
  const hPos     = el("hud-pos");
  const hTime    = el("hud-time");
  const hDust    = el("hud-dust");
  const hPlayers = el("hud-players");
  const status   = el("status");

  if (hWorld)   hWorld.textContent   = `world: ${world.worldId || "—"}`;
  if (hPos)     hPos.textContent     = `pos: ${player.pos.map(v => v.toFixed(1)).join(", ")}`;
  if (hTime)    hTime.textContent    = `time: ${world.simTime.toFixed(1)}s`;
  if (hDust)    hDust.textContent    = `dust: ${world.globalDust.toFixed(3)}`;
  if (hPlayers) hPlayers.textContent = `players: ${remotePlayers.size + 1}`;

  if (status) {
    status.className = _stage === Stage.Running  ? "connected"  :
                       _stage === Stage.LoadMinimal ? "disconnected" : "connecting";
    status.textContent = _stage;
  }
}

setInterval(_updateHUD, 250);

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
  console.info("[Dust] Boot — Stage 23");

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
        type:         "REJOIN_RESYNC",
        lastSeenTick: _lastSeenEpoch,
        lastEventId:  _lastEventId,
        lastPatchId:  0,
      });
    }
  }, 30_000);
}

_boot();
