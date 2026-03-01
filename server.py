"""server.py — Dust Stage 21 multiplayer server entry point.

Usage
-----
    python server.py [--seed SEED] [--reset] [--port PORT] [--state-dir DIR]

Opens http://localhost:8765/ in a browser to join the shared world.

On first run a fresh world is created in ``world_state/``.
To reset the world: delete ``world_state/`` (or pass ``--reset``).
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

_TAG = "Server"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dust multiplayer server (Stage 21)",
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
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    Logger.set_level(LogLevel.INFO)
    config = Config()

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
    world_state = WorldState(args.state_dir)
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
        state_dir    = args.state_dir,
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
