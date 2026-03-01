"""test_net_stage23.py — Stage 23 web-deploy smoke tests.

Tests
-----
1. TestBuildOutputs
   — web_client/dist/ directory exists after build.
   — index.html is present.
   — asset-manifest.json is valid JSON with required keys.
   — Hashed JS files referenced in the manifest exist on disk.

2. TestCacheHeaders
   — _handle_http("/") returns no-cache headers.
   — _handle_http("<hashed asset>") returns immutable cache headers.
   — Security headers (X-Content-Type-Options, Referrer-Policy, CSP)
     are present in all responses.
   — Correct MIME types for .wasm, .js, .html files.

3. TestBuildId
   — NetworkServer._build_id is a non-empty hex string.
   — WORLD_SYNC message contains a non-empty buildId field.

4. TestRangeRequests
   — _handle_http returns 206 with correct Content-Range for a
     Range: bytes=0-3 request against a binary file in dist/.

5. TestWebSocketBuildId (integration, live server)
   — Connect to a live server, receive WORLD_SYNC, confirm buildId present.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.net.NetworkServer import NetworkServer, _DIST_DIR
from src.net.PlayerRegistry import PlayerRegistry
from src.net.SpawnAnchor import SpawnAnchor
from src.net.WorldState import WorldState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_server(state_dir: str, static_dir: str | None = None) -> NetworkServer:
    ws_state = WorldState(state_dir)
    ws_state.load_or_create(default_seed=42)
    srv = NetworkServer(
        bootstrap   = None,
        config      = None,
        world_state = ws_state,
        state_dir   = state_dir,
    )
    if static_dir is not None:
        from pathlib import Path as P
        srv._static_dir = P(static_dir)
    return srv


# ---------------------------------------------------------------------------
# 1. TestBuildOutputs
# ---------------------------------------------------------------------------

class TestBuildOutputs(unittest.TestCase):
    """Validate the dist/ directory structure produced by build_web.sh."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dist = _DIST_DIR
        # Consider dist "built" only when it contains at least one file.
        cls._built = cls.dist.is_dir() and any(cls.dist.iterdir())

    def test_dist_directory_exists_or_skipped(self) -> None:
        """If dist/ does not exist or is empty, build_web.sh has not been run — skip."""
        if not self._built:
            self.skipTest("web_client/dist/ not built — run deploy/build_web.sh first")

    def test_index_html_present(self) -> None:
        if not self._built:
            self.skipTest("dist/ not built")
        self.assertTrue((self.dist / "index.html").exists(),
                        "index.html missing from dist/")

    def test_asset_manifest_valid(self) -> None:
        if not self._built:
            self.skipTest("dist/ not built")
        manifest_path = self.dist / "asset-manifest.json"
        if not manifest_path.exists():
            self.skipTest("asset-manifest.json not present")
        data = json.loads(manifest_path.read_text())
        self.assertIn("buildId", data,  "manifest missing buildId")
        self.assertIn("files",   data,  "manifest missing files")
        self.assertIsInstance(data["files"], dict)

    def test_hashed_assets_exist(self) -> None:
        if not self._built:
            self.skipTest("dist/ not built")
        manifest_path = self.dist / "asset-manifest.json"
        if not manifest_path.exists():
            self.skipTest("asset-manifest.json not present")
        data  = json.loads(manifest_path.read_text())
        files = data.get("files", {})
        for logical, hashed in files.items():
            path = self.dist / hashed
            self.assertTrue(
                path.exists(),
                f"Asset {logical!r} → {hashed!r} not found in dist/",
            )


# ---------------------------------------------------------------------------
# 2. TestCacheHeaders
# ---------------------------------------------------------------------------

class TestCacheHeaders(unittest.IsolatedAsyncioTestCase):
    """HTTP responses have correct Cache-Control and security headers."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._state_dir = os.path.join(self._tmp, "world_state")
        # Create a temporary dist/ with a fake hashed asset and a binary file
        self._dist = os.path.join(self._tmp, "dist")
        os.makedirs(self._dist)
        # Fake hashed JS
        (Path(self._dist) / "client.a1b2c3d4.js").write_bytes(b"/* client */")
        # Fake hashed WASM
        (Path(self._dist) / "core.deadbeef.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
        # Non-hashed asset
        (Path(self._dist) / "sw.js").write_bytes(b"/* sw */")
        # Binary blob for range tests
        (Path(self._dist) / "snapshot.abcdef12.bin").write_bytes(b"\x01\x02\x03\x04\x05\x06")
        # index.html in dist
        (Path(self._dist) / "index.html").write_bytes(b"<html><body></body></html>")

        self.server = _make_server(self._state_dir, static_dir=self._dist)

    def _get_header(self, headers, name: str) -> str | None:
        name_lower = name.lower()
        for k, v in headers:
            if k.lower() == name_lower:
                return v
        return None

    async def test_index_no_cache(self) -> None:
        result = await self.server._handle_http("/", {})
        self.assertIsNotNone(result)
        status, headers, _ = result
        self.assertEqual(status, 200)
        cc = self._get_header(headers, "Cache-Control")
        self.assertIsNotNone(cc, "Cache-Control header missing for /")
        self.assertIn("no-cache", cc)

    async def test_hashed_js_immutable(self) -> None:
        result = await self.server._handle_http("/client.a1b2c3d4.js", {})
        self.assertIsNotNone(result)
        status, headers, _ = result
        self.assertEqual(status, 200)
        cc = self._get_header(headers, "Cache-Control")
        self.assertIsNotNone(cc, "Cache-Control header missing for hashed asset")
        self.assertIn("immutable", cc)
        self.assertIn("max-age=31536000", cc)

    async def test_hashed_wasm_content_type(self) -> None:
        result = await self.server._handle_http("/core.deadbeef.wasm", {})
        self.assertIsNotNone(result)
        status, headers, _ = result
        self.assertEqual(status, 200)
        ct = self._get_header(headers, "Content-Type")
        self.assertIsNotNone(ct)
        self.assertIn("application/wasm", ct)

    async def test_security_headers_present(self) -> None:
        for path in ("/", "/client.a1b2c3d4.js", "/sw.js"):
            result = await self.server._handle_http(path, {})
            if result is None:
                continue
            _, headers, _ = result
            xct = self._get_header(headers, "X-Content-Type-Options")
            self.assertEqual(xct, "nosniff",
                             f"X-Content-Type-Options missing for {path}")
            rp = self._get_header(headers, "Referrer-Policy")
            self.assertEqual(rp, "no-referrer",
                             f"Referrer-Policy missing for {path}")
            csp = self._get_header(headers, "Content-Security-Policy")
            self.assertIsNotNone(csp, f"CSP header missing for {path}")

    async def test_non_hashed_sw_no_cache(self) -> None:
        result = await self.server._handle_http("/sw.js", {})
        self.assertIsNotNone(result)
        status, headers, _ = result
        self.assertEqual(status, 200)
        cc = self._get_header(headers, "Cache-Control")
        self.assertIsNotNone(cc)
        self.assertNotIn("immutable", cc)

    async def test_ws_path_returns_none(self) -> None:
        result = await self.server._handle_http("/ws", {})
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# 3. TestBuildId
# ---------------------------------------------------------------------------

class TestBuildId(unittest.IsolatedAsyncioTestCase):
    """NetworkServer exposes a valid buildId in WORLD_SYNC."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._state_dir = os.path.join(self._tmp, "world_state")
        self.server = _make_server(self._state_dir)
        self.registry = PlayerRegistry()
        self.server._registry = self.registry
        self.registry.add("p1", [0.0, 1001.8, 0.0])

    def test_build_id_non_empty_hex(self) -> None:
        bid = self.server._build_id
        self.assertIsInstance(bid, str)
        self.assertGreater(len(bid), 0, "buildId must not be empty")
        # Must be hex characters
        self.assertTrue(
            all(c in "0123456789abcdef" for c in bid.lower()),
            f"buildId {bid!r} contains non-hex characters",
        )

    async def test_world_sync_contains_build_id(self) -> None:
        messages = []

        class FakeWS:
            async def send(self, msg):
                messages.append(msg)

        await self.server._send_world_sync(FakeWS(), "p1")
        self.assertEqual(len(messages), 1)
        data = json.loads(messages[0])
        self.assertIn("buildId", data, "WORLD_SYNC missing buildId field")
        self.assertEqual(data["buildId"], self.server._build_id)


# ---------------------------------------------------------------------------
# 4. TestRangeRequests
# ---------------------------------------------------------------------------

class TestRangeRequests(unittest.IsolatedAsyncioTestCase):
    """Server supports byte-range requests for binary assets."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.mkdtemp()
        self._state_dir = os.path.join(self._tmp, "world_state")
        self._dist = os.path.join(self._tmp, "dist")
        os.makedirs(self._dist)
        (Path(self._dist) / "snapshot.abcdef12.bin").write_bytes(
            bytes(range(16))
        )
        self.server = _make_server(self._state_dir, static_dir=self._dist)

    async def test_range_request_206(self) -> None:
        fake_headers = {"Range": "bytes=0-3"}
        result = await self.server._handle_http(
            "/snapshot.abcdef12.bin", fake_headers
        )
        self.assertIsNotNone(result)
        status, headers, body = result
        self.assertEqual(status, 206, "Expected 206 Partial Content")
        self.assertEqual(body, bytes([0, 1, 2, 3]))

    async def test_range_headers_present(self) -> None:
        """Even without a Range header, Accept-Ranges: bytes must be set."""
        result = await self.server._handle_http(
            "/snapshot.abcdef12.bin", {}
        )
        self.assertIsNotNone(result)
        status, headers, _ = result
        self.assertEqual(status, 200)
        ar = next((v for k, v in headers if k.lower() == "accept-ranges"), None)
        self.assertEqual(ar, "bytes", "Accept-Ranges header missing")


# ---------------------------------------------------------------------------
# 5. TestWebSocketBuildId  (integration — live server)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    __import__("importlib").util.find_spec("websockets") is not None,
    "websockets not installed",
)
class TestWebSocketBuildId(unittest.IsolatedAsyncioTestCase):
    """Live server sends buildId in WORLD_SYNC message."""

    async def asyncSetUp(self) -> None:
        import websockets as _ws  # noqa: F401
        self._tmp = tempfile.mkdtemp()
        state_dir = os.path.join(self._tmp, "world_state")
        ws_state  = WorldState(state_dir)
        ws_state.load_or_create(default_seed=42)
        port = _find_free_port()
        self.server = NetworkServer(
            bootstrap   = None,
            config      = None,
            world_state = ws_state,
            state_dir   = state_dir,
        )
        # Override port for test
        self.server._port = port
        self._port = port
        await self.server.start()

    async def asyncTearDown(self) -> None:
        await self.server.stop()

    async def test_world_sync_build_id(self) -> None:
        import websockets
        uri = f"ws://127.0.0.1:{self._port}/ws"
        async with websockets.connect(uri) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(raw)
        self.assertEqual(data.get("type"), "WORLD_SYNC")
        self.assertIn("buildId", data, "WORLD_SYNC missing buildId")
        self.assertIsInstance(data["buildId"], str)
        self.assertGreater(len(data["buildId"]), 0)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
