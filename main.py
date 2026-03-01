"""main.py — entry point for Dust Stage 2: Planet Walk prototype."""
import sys
import os

# Ensure src is on path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.core.EngineCore import EngineCore


def main() -> None:
    engine = EngineCore()
    try:
        engine.init()
        engine.run()
    except KeyboardInterrupt:
        pass
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
