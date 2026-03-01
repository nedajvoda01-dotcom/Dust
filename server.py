"""server.py — Dust Stage 24 multiplayer server entry point.

Usage
-----
    python server.py [--seed SEED] [--reset] [--port PORT] [--state-dir DIR]

Opens http://localhost:8765/ in a browser to join the shared world.

On first run a fresh world is created in ``world_state/``.
To reset the world: delete ``world_state/`` (or pass ``--reset``).

Stage 24 ops commands (no server needed):
    python server.py --status            — print world status and health
    python server.py --compact-now       — compact world_state/ and exit
    python server.py --reset-world-soft  — create RESET_NOW flag and exit
    python server.py --prune-logs        — prune old log files and exit
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from src.core.Config import Config
from src.core.GameBootstrap import GameBootstrap
from src.core.Logger import Logger, LogLevel
from src.net.NetworkServer import NetworkServer
from src.net.WorldState import WorldState
from src.ops.OpsLayer import OpsLayer

_TAG = "Server"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dust multiplayer server (Stage 24)",
    )
    parser.add_argument("--seed",      type=int,   default=None,
                        help="World seed (overrides saved world seed on --reset)")
    parser.add_argument("--reset",     action="store_true",
                        help="Delete world_state/ and start a fresh world")
    parser.add_argument("--port",      type=int,   default=None,
                        help="WebSocket port (overrides config net.ws_port)")
    parser.add_argument("--state-dir", type=str,   default="world_state",
                        help="Path to the world_state/ directory")
    parser.add_argument("--headless",  action="store_true",
                        help="Run simulation headlessly (no display/audio)")
    # Stage 24 — ops commands (run without starting the server)
    parser.add_argument("--status",           action="store_true",
                        help="Print world status (health) and exit")
    parser.add_argument("--compact-now",      action="store_true",
                        help="Compact world_state/ (create baseline + prune) and exit")
    parser.add_argument("--reset-world-soft", action="store_true",
                        help="Create RESET_NOW flag for a running server and exit")
    parser.add_argument("--prune-logs",       action="store_true",
                        help="Prune old rotated log files and exit")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    Logger.set_level(LogLevel.INFO)
    config = Config()

    state_dir = args.state_dir

    # ------------------------------------------------------------------
    # Stage 24 — Ops-only commands (no server, no bootstrap)
    # ------------------------------------------------------------------
    if args.status or args.compact_now or args.reset_world_soft or args.prune_logs:
        world_state = WorldState(state_dir)
        world_state.load_or_create(default_seed=42)
        ops = OpsLayer(world_state=world_state, config=config, state_dir=state_dir)

        if args.status:
            import json as _json
            print(_json.dumps(ops.health(), indent=2))

        if args.compact_now:
            ok = ops.compact()
            print("compact-now:", "OK" if ok else "FAILED")

        if args.reset_world_soft:
            flag = ops._reset_flag_path
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.touch()
            print(f"reset-world-soft: flag created at {flag}")

        if args.prune_logs:
            ops.prune_logs()
            print("prune-logs: OK")
        return

    # Override port if provided via CLI
    if args.port is not None:
        # Monkey-patch config for the NetworkServer to pick up
        # (simplest approach without modifying Config internals)
        original_get = config.get

        def patched_get(section, key, *rest, **kwargs):
            if section == "net" and key == "ws_port":
                return args.port
            return original_get(section, key, *rest, **kwargs)

        config.get = patched_get  # type: ignore[method-assign]

    # WorldState — load or create
    world_state = WorldState(state_dir)
    default_seed = args.seed if args.seed is not None else int(
        config.get("seed", "default", default=42)
    )
    if args.reset:
        Logger.info(_TAG, "--reset: wiping world_state/")
        world_state.reset()
        world_state.seed = default_seed
        world_state.save()
    else:
        world_state.load_or_create(default_seed=default_seed)

    # Bootstrap the simulation
    bootstrap_args = ["--headless"]
    if args.seed is not None:
        bootstrap_args += ["--seed", str(args.seed)]
    if args.reset:
        bootstrap_args += ["--reset"]

    bootstrap = GameBootstrap(headless=True)  # server always headless
    bootstrap.init(cli_args=bootstrap_args)

    # Sync seed into world state from the bootstrap identity
    world_state.seed     = bootstrap.seed
    world_state.world_id = bootstrap.identity.world_id
    world_state.save()

    # NetworkServer
    server = NetworkServer(
        bootstrap    = bootstrap,
        config       = config,
        world_state  = world_state,
        state_dir    = state_dir,
    )

    try:
        await server.serve_forever()
    finally:
        bootstrap.shutdown()
        Logger.info(_TAG, "Server stopped")


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
