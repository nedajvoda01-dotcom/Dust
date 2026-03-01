"""Logger — levelled console logger. No UI, stdout/stderr only."""
import sys
import time
from enum import IntEnum


class LogLevel(IntEnum):
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3


class Logger:
    _level: LogLevel = LogLevel.INFO
    _start: float = time.monotonic()

    @classmethod
    def set_level(cls, level: LogLevel) -> None:
        cls._level = level

    @classmethod
    def _log(cls, level: LogLevel, tag: str, msg: str) -> None:
        if level < cls._level:
            return
        elapsed = time.monotonic() - cls._start
        label = level.name
        line = f"[{elapsed:8.3f}] [{label:5s}] [{tag}] {msg}"
        out = sys.stderr if level >= LogLevel.WARN else sys.stdout
        print(line, file=out)

    @classmethod
    def debug(cls, tag: str, msg: str) -> None:
        cls._log(LogLevel.DEBUG, tag, msg)

    @classmethod
    def info(cls, tag: str, msg: str) -> None:
        cls._log(LogLevel.INFO, tag, msg)

    @classmethod
    def warn(cls, tag: str, msg: str) -> None:
        cls._log(LogLevel.WARN, tag, msg)

    @classmethod
    def error(cls, tag: str, msg: str) -> None:
        cls._log(LogLevel.ERROR, tag, msg)
