"""API keys for TTS servers, one per server provider. They live in their own file, readable only by the owner,
and never in the settings file, so ~/.epub2audiobook.json can be shared (e.g. in a bug report) without them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

KEYS_FILE = Path.home() / ".epub2audiobook-keys.json"


def load_keys() -> dict[str, str]:
    """Saved keys by provider ({} if there is no readable file)."""
    try:
        data = json.loads(KEYS_FILE.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {p: k for p, k in data.items() if isinstance(k, str) and k}


def save_keys(keys: dict[str, str]) -> None:
    """Write the non-empty keys with permission 600; with no key left, remove the file."""
    keys = {p: k for p, k in keys.items() if k}
    if not keys:
        KEYS_FILE.unlink(missing_ok=True)
        return
    # the new file gets 600 before the key is in it; os.replace keeps that (an old file could have wider rights)
    tmp = KEYS_FILE.with_name(KEYS_FILE.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(tmp, 0o600)  # O_CREAT's mode does not apply when the file already exists
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=2)
    os.replace(tmp, KEYS_FILE)
