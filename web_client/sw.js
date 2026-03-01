/**
 * sw.js — Dust Stage 23 Service Worker.
 *
 * Caches all hashed static assets on install.
 * On activate, purges caches from previous builds.
 * Posts SW_UPDATED to all clients when a new version activates,
 * prompting a location.reload() (with debounce).
 *
 * Cache strategy:
 *   - Hashed assets (filename contains an 8+ hex segment):
 *       cache-first (they are immutable).
 *   - index.html and non-hashed files:
 *       network-first, fall back to cache.
 */
"use strict";

// SW_VERSION is replaced by build_web.sh with the actual build ID.
const SW_VERSION  = "__BUILD_ID__";
const CACHE_NAME  = `dust-${SW_VERSION}`;

// Hashed asset pattern (matches names like client.a1b2c3d4.js)
const HASHED_RE   = /\.[0-9a-f]{8,}\./i;

// ---------------------------------------------------------------------------
// Install — precache known hashed assets
// ---------------------------------------------------------------------------
self.addEventListener("install", (ev) => {
  // Skip waiting so the new SW activates immediately.
  self.skipWaiting();
  // The actual asset list is populated by build_web.sh writing an
  // asset-manifest.json; here we cache what we can fetch.
  ev.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      cache.addAll(["/"])
    )
  );
});

// ---------------------------------------------------------------------------
// Activate — prune stale caches
// ---------------------------------------------------------------------------
self.addEventListener("activate", (ev) => {
  ev.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k.startsWith("dust-") && k !== CACHE_NAME)
          .map((k) => caches.delete(k))
      )
    ).then(() => {
      // Claim all open clients immediately.
      return self.clients.claim();
    }).then(() => {
      // Notify all clients that a new version is active.
      return self.clients.matchAll({ type: "window" }).then((clients) => {
        clients.forEach((c) => c.postMessage({ type: "SW_UPDATED" }));
      });
    })
  );
});

// ---------------------------------------------------------------------------
// Fetch — cache-first for hashed assets, network-first for everything else
// ---------------------------------------------------------------------------
self.addEventListener("fetch", (ev) => {
  const url = new URL(ev.request.url);

  // Never intercept WebSocket upgrades or cross-origin requests.
  if (url.protocol === "ws:" || url.protocol === "wss:") return;
  if (url.origin !== location.origin) return;

  const isHashed = HASHED_RE.test(url.pathname);

  if (isHashed) {
    // Cache-first: hashed assets are immutable.
    ev.respondWith(
      caches.match(ev.request).then((cached) => {
        if (cached) return cached;
        return fetch(ev.request).then((resp) => {
          if (resp && resp.status === 200) {
            const clone = resp.clone();
            caches.open(CACHE_NAME).then((c) => c.put(ev.request, clone));
          }
          return resp;
        });
      })
    );
  } else {
    // Network-first: always try to get the latest index.html etc.
    ev.respondWith(
      fetch(ev.request)
        .then((resp) => {
          if (resp && resp.status === 200) {
            const clone = resp.clone();
            caches.open(CACHE_NAME).then((c) => c.put(ev.request, clone));
          }
          return resp;
        })
        .catch(() => caches.match(ev.request))
    );
  }
});
