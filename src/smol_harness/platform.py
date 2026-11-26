"""Operating-system detection and per-platform capability.

Centralises the answer to "what OS are we on?" so the harness does not scatter
``platform.system()`` calls through tool code, and so Windows (and other
non-Linux hosts) get a clear, graceful failure message instead of an obscure
``AttributeError`` on a POSIX-only flag.
"""

from __future__ import annotations

import os
import platform
import shutil

# Canonical OS name, as ``platform.system()`` reports it: "Linux", "Windows", "Darwin", "Java".
OS_NAME = platform.system()
OS_RELEASE = platform.release()
OS_MACHINE = platform.machine()


def is_windows() -> bool:
    """True on Microsoft Windows. WSL2 reports "Linux", so this is False there."""
    return OS_NAME == "Windows"


def is_linux() -> bool:
    return OS_NAME == "Linux"


def is_macos() -> bool:
    return OS_NAME == "Darwin"


def os_display_name() -> str:
    """A human-readable "OS release (machine)" string, e.g. "Windows 10 (AMD64)". """
    return f"{OS_NAME} {OS_RELEASE} ({OS_MACHINE})"


def shell_available() -> bool:
    """Whether effGen's Linux-namespace shell sandbox can be enforced here.

    Needs Linux, plus ``bash``, ``unshare`` and GNU ``timeout`` on PATH.
    Windows and macOS return False.
    """
    if not is_linux():
        return False
    return all(shutil.which(binary) is not None for binary in ("bash", "unshare", "timeout"))


def shell_unavailable_reason() -> str | None:
    """Why the shell tool cannot be built on this host, or None if it can."""
    if shell_available():
        return None
    if is_windows():
        return (
            "the shell tool needs Linux (unprivileged user namespaces, bash, unshare, GNU "
            "timeout). On Windows, run the harness inside WSL2 (not tested)."
        )
    return f"this host ({os_display_name()}) cannot enforce the shell sandbox"
