"""PlayerIdentity — deterministic player key from IP + User-Agent + server salt.

The player key is never shown in UI; it is used purely as a stable
internal identifier so the same browser session reconnects as the same
"player" without any account system.

Formula
-------
    key = sha256(remote_ip + "|" + user_agent + "|" + server_salt)[:16]

The ``server_salt`` is generated fresh each server start, so keys do not
persist across restarts (by design — no accounts).
"""
from __future__ import annotations

import hashlib
import secrets

_DEFAULT_SALT: str | None = None


def _get_default_salt() -> str:
    global _DEFAULT_SALT
    if _DEFAULT_SALT is None:
        _DEFAULT_SALT = secrets.token_hex(16)
    return _DEFAULT_SALT


def make_player_key(
    remote_ip: str,
    user_agent: str,
    server_salt: str | None = None,
) -> str:
    """Return a stable 16-hex-char player key for *remote_ip* + *user_agent*.

    Parameters
    ----------
    remote_ip:
        The client's remote IP address string (IPv4 or IPv6).
    user_agent:
        The HTTP ``User-Agent`` header sent by the client.
    server_salt:
        Per-server-run random token.  If *None* a module-level default
        salt is generated and reused for the process lifetime.
    """
    if server_salt is None:
        server_salt = _get_default_salt()
    payload = f"{remote_ip}|{user_agent}|{server_salt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
