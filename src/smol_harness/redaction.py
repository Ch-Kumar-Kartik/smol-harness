"""The one redaction policy for anything written to shared or archived evidence.

Standard library only, so that `experiments/rtx4070/rtx4070.py` (which runs before the project environment exists) can load
this file by path, and `recorder.py` can import it normally. `redact` and `redact_obj` were moved verbatim from the experiment
driver; `redact_record` (values, then text) is how both writers turn a record into a line. Do not weaken it here for one caller.
"""

from __future__ import annotations

import getpass
import json
import os
import re
from pathlib import Path

_SECRET_NAME = re.compile(r"(?i)(key|token|secret|passw|credential|auth|cookie)")
# A key block a model happened to `cat` (shell reads are not confined, R1): the whole block goes, including a truncated one.
_PEM = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S)
_REDACTIONS = (
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "hf_***"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "gh_***"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "sk-***"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer ***"),
    (re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)[^/\s:@]+(?::[^/\s@]*)?@"), r"\g<scheme>***@"),
    (re.compile(r"(?i)([\w.-]*(?:api[_-]?key|token|password|passwd|secret|credential)[\w.-]*)(\s*[=:]\s*)[^\s,;\"']+"), r"\1\2***"),
    (_PEM, "-----BEGIN PRIVATE KEY (redacted)-----"),
)


def redact(text: str) -> str:
    """Remove credentials and personal paths from text before it is written anywhere."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    for name, value in os.environ.items():  # exact values of sensitive environment variables, wherever they appear
        if _SECRET_NAME.search(name) and len(value) >= 8:
            text = text.replace(value, "***")
    try:
        user, home = getpass.getuser(), str(Path.home())
    except Exception:  # noqa: BLE001 - no resolvable user: nothing personal to remove
        return text
    if home not in ("/", ""):
        text = text.replace(home, "~")
    return re.sub(r"/mnt/c/Users/[^/\s]+", "/mnt/c/Users/<user>", text.replace("/home/" + user, "~")) if user else text


def redact_record(record, *, ensure_ascii: bool = True) -> str:
    """One JSON line for a record, redacted. Values are redacted BEFORE serialising: `json.dumps` escapes quotes, backslashes and control
    characters, so a secret containing them no longer matches its exact environment value in the serialised text (Opus review of history
    Phase 2, B2). The text pass then runs again over the line as a second defence (patterns spanning fields); if it ever broke the JSON, the
    value-redacted line is kept. Non-JSON values are turned into strings first (`default=str`), so they are redacted too."""
    plain = json.loads(json.dumps(record, ensure_ascii=False, default=str))
    line = json.dumps(redact_obj(plain), ensure_ascii=ensure_ascii)
    text = redact(line)
    try:
        json.loads(text)
    except ValueError:
        return line
    return text


def redact_obj(obj):
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: ("***" if isinstance(k, str) and _SECRET_NAME.search(k) and isinstance(v, str) and len(v) >= 8 else redact_obj(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj
