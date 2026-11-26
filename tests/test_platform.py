"""Tests for `smol_harness.platform`: OS auto-detection and capability reporting."""

from __future__ import annotations

from smol_harness.platform import OS_NAME, is_linux, is_macos, is_windows, os_display_name


def test_os_name_is_a_plausible_value() -> None:
    """`platform.system()` returns one of the known OS names."""
    assert OS_NAME in {"Linux", "Windows", "Darwin", "Java"}


def test_exactly_one_os_flag_true() -> None:
    """Exactly one of is_windows / is_linux / is_macos is True on any host."""
    flags = [is_windows(), is_linux(), is_macos()]
    assert sum(flags) == 1


def test_os_display_name_contains_os_name() -> None:
    """`os_display_name()` includes the canonical OS name."""
    assert OS_NAME in os_display_name()


def test_shell_available_matches_linux() -> None:
    """On non-Linux hosts `shell_available()` is False (it can only be True on Linux)."""
    from smol_harness.platform import shell_available

    if not is_linux():
        assert shell_available() is False
