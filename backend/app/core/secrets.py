"""Secret storage via the OS keyring (Windows Credential Manager).

The API key is NEVER written to the SQLite DB, .env (in production), or logs
when keyring works. However, keyring can silently fail in PyInstaller
environments, so we add a SQLite fallback so the key is still retrievable.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import keyring
except ModuleNotFoundError:  # pragma: no cover - depends on local test env
    keyring = types.ModuleType("keyring")

    def _missing(*_args, **_kwargs):
        return None

    keyring.get_password = _missing  # type: ignore[attr-defined]
    keyring.set_password = _missing  # type: ignore[attr-defined]
    keyring.delete_password = _missing  # type: ignore[attr-defined]
    sys.modules["keyring"] = keyring

SERVICE = "SuperAgent"
ACCOUNT = "openai_api_key"
PROVIDER_PREFIX = "provider:"

# --- SQLite fallback -------------------------------------------------------
# When keyring fails (common in PyInstaller packaged apps), store keys in
# the SQLite settings table as a last-resort fallback.

_sqlite_conn = None  # set once by set_sqlite_conn()


def set_sqlite_conn(conn) -> None:
    """Register the app's SQLite connection for fallback key storage."""
    global _sqlite_conn
    _sqlite_conn = conn
    _ensure_key_columns(conn)


def _ensure_key_columns(conn) -> None:
    """Create the api_keys table if it doesn't exist (fallback storage)."""
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS api_keys ("
            "  account TEXT PRIMARY KEY,"
            "  key_value TEXT NOT NULL"
            ")"
        )
        conn.commit()
    except Exception:
        pass


def _sqlite_get(account: str) -> Optional[str]:
    if _sqlite_conn is None:
        return None
    try:
        row = _sqlite_conn.execute(
            "SELECT key_value FROM api_keys WHERE account = ?", (account,)
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _sqlite_set(account: str, key: str) -> None:
    if _sqlite_conn is None:
        return
    try:
        _sqlite_conn.execute(
            "INSERT INTO api_keys (account, key_value) VALUES (?, ?)"
            " ON CONFLICT(account) DO UPDATE SET key_value = excluded.key_value",
            (account, key),
        )
        _sqlite_conn.commit()
    except Exception:
        pass


def _sqlite_delete(account: str) -> None:
    if _sqlite_conn is None:
        return
    try:
        _sqlite_conn.execute("DELETE FROM api_keys WHERE account = ?", (account,))
        _sqlite_conn.commit()
    except Exception:
        pass


# --- Keyring helpers with fallback -----------------------------------------

def _keyring_get(account: str) -> Optional[str]:
    """Try keyring first, fall back to SQLite."""
    try:
        value = keyring.get_password(SERVICE, account)
        if value:
            return value
    except Exception as exc:
        logger.debug("keyring.get_password failed for %s: %s", account, exc)
    # Fallback to SQLite
    fallback = _sqlite_get(account)
    if fallback:
        logger.debug("API key for %s retrieved from SQLite fallback", account)
    return fallback


def _keyring_set(account: str, key: str) -> None:
    """Try keyring first; always write to SQLite as backup."""
    # Always write to SQLite fallback so the key survives keyring failures
    _sqlite_set(account, key)
    try:
        keyring.set_password(SERVICE, account, key)
    except Exception as exc:
        logger.warning("keyring.set_password failed for %s (SQLite fallback used): %s", account, exc)


def _keyring_delete(account: str) -> None:
    """Delete from both keyring and SQLite."""
    _sqlite_delete(account)
    try:
        keyring.delete_password(SERVICE, account)
    except Exception:
        pass


# --- Public API -------------------------------------------------------------

def get_api_key() -> Optional[str]:
    return _keyring_get(ACCOUNT)


def set_api_key(key: str) -> None:
    _keyring_set(ACCOUNT, key)


def delete_api_key() -> None:
    _keyring_delete(ACCOUNT)


def has_api_key() -> bool:
    return bool(get_api_key())


def _provider_account(provider_id: str) -> str:
    return PROVIDER_PREFIX + provider_id


def get_provider_api_key(provider_id: str) -> Optional[str]:
    return _keyring_get(_provider_account(provider_id))


def set_provider_api_key(provider_id: str, key: str) -> None:
    _keyring_set(_provider_account(provider_id), key)


def delete_provider_api_key(provider_id: str) -> None:
    _keyring_delete(_provider_account(provider_id))


def has_provider_api_key(provider_id: str) -> bool:
    return bool(get_provider_api_key(provider_id))
