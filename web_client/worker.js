/**
 * worker.js — Dust Stage 23 network worker.
 *
 * Runs in a dedicated Web Worker to keep the main thread light.
 * Responsibilities:
 *   - Own the WebSocket connection and reconnect logic.
 *   - Decode incoming server messages (JSON) and post structured
 *     events to the main thread.
 *   - Accept SEND commands from the main thread and forward them
 *     over the WebSocket.
 *
 * Messages TO main thread (postMessage):
 *   { type: "WS_OPEN" }
 *   { type: "WS_CLOSE", code, reason }
 *   { type: "WS_MSG",  data: <parsed JSON object> }
 *   { type: "WS_ERROR", message }
 *
 * Messages FROM main thread (onmessage):
 *   { type: "CONNECT",  url: <ws url> }
 *   { type: "SEND",     payload: <JSON string or object> }
 *   { type: "DISCONNECT" }
 */
"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let _ws         = null;
let _url        = null;
let _reconnectN = 0;
let _reconnectTimer = null;

// Exponential back-off: 500ms → 1s → 2s → 5s → 10s (capped)
const _BACKOFF = [500, 1000, 2000, 5000, 10000];

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
function _connect(url) {
  _url = url;
  if (_ws) {
    try { _ws.close(); } catch (_) {}
    _ws = null;
  }

  try {
    _ws = new WebSocket(url);
    _ws.binaryType = "arraybuffer";

    _ws.onopen = () => {
      _reconnectN = 0;
      postMessage({ type: "WS_OPEN" });
    };

    _ws.onclose = (ev) => {
      _ws = null;
      postMessage({ type: "WS_CLOSE", code: ev.code, reason: ev.reason });
      _scheduleReconnect();
    };

    _ws.onerror = () => {
      postMessage({ type: "WS_ERROR", message: "WebSocket error" });
    };

    _ws.onmessage = (ev) => {
      try {
        const parsed = JSON.parse(ev.data);
        postMessage({ type: "WS_MSG", data: parsed });
      } catch (e) {
        postMessage({ type: "WS_ERROR", message: "JSON parse error" });
      }
    };
  } catch (e) {
    postMessage({ type: "WS_ERROR", message: String(e) });
    _scheduleReconnect();
  }
}

function _scheduleReconnect() {
  if (_reconnectTimer !== null) return;
  const delay = _BACKOFF[Math.min(_reconnectN, _BACKOFF.length - 1)];
  _reconnectN += 1;
  _reconnectTimer = setTimeout(() => {
    _reconnectTimer = null;
    if (_url) _connect(_url);
  }, delay);
}

// ---------------------------------------------------------------------------
// Message handler (commands from main thread)
// ---------------------------------------------------------------------------
self.onmessage = (ev) => {
  const msg = ev.data;
  if (!msg || !msg.type) return;

  switch (msg.type) {
    case "CONNECT":
      _reconnectN = 0;
      if (_reconnectTimer !== null) {
        clearTimeout(_reconnectTimer);
        _reconnectTimer = null;
      }
      _connect(msg.url);
      break;

    case "SEND":
      if (_ws && _ws.readyState === WebSocket.OPEN) {
        const payload = typeof msg.payload === "string"
          ? msg.payload
          : JSON.stringify(msg.payload);
        _ws.send(payload);
      }
      break;

    case "DISCONNECT":
      _reconnectN = 0;
      if (_reconnectTimer !== null) {
        clearTimeout(_reconnectTimer);
        _reconnectTimer = null;
      }
      _url = null;
      if (_ws) {
        try { _ws.close(1000, "client disconnect"); } catch (_) {}
        _ws = null;
      }
      break;

    default:
      break;
  }
};
