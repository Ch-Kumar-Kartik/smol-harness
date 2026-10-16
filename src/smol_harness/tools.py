"""Per-agent tool factory: fresh effGen tool instances for exactly the allowlisted names.

The registry is fixed. Web or network tools, `delegate` (the harness adds it to the
orchestrator only) and unknown names are never built. Path confinement for
`file_operations` is effGen's own (`FileOperations`, audit §1.9); this module only
validates the workspace and checks that effGen kept it as the confinement root.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

from effgen.tools.base_tool import BaseTool
from effgen.tools.builtin.file_ops import FileOperations

from smol_harness.delegation import DELEGATE_TOOL_NAME


class ToolBuildError(ValueError):
    """A tool cannot be built: unknown/forbidden name, bad workspace, or unavailable tool."""


def resolve_workspace(workspace: Path | str) -> Path:
    """Return the resolved workspace; it must be an existing, accessible directory."""
    if not str(workspace).strip():  # Path("") would silently mean the current directory
        raise ToolBuildError("workspace must be a non-empty path")
    try:
        path = Path(workspace).resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise ToolBuildError(f"workspace {str(workspace)!r} does not exist or cannot be resolved: {e}") from e
    if not path.is_dir():
        raise ToolBuildError(f"workspace {str(path)!r} is not a directory")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        raise ToolBuildError(f"workspace {str(path)!r} is not readable, writable and searchable by this process")
    return path


def _file_operations(workspace: Path) -> BaseTool:
    tool = FileOperations(allowed_directories=[str(workspace)])
    # effGen silently replaces a missing allowed directory with EFFGEN_WORKSPACE or the
    # current directory (audit §1.9, T4.1). Read back the root it kept so a workspace that
    # vanished after validation fails closed instead of widening confinement. The attribute
    # is private; there is no public accessor.
    if tool._allowed_directories != [workspace]:
        raise ToolBuildError(
            f"file_operations did not confine to {str(workspace)!r} (effGen kept "
            f"{[str(p) for p in tool._allowed_directories]}); refusing to build it"
        )
    return tool


def _shell(workspace: Path) -> BaseTool:
    # Fail closed: never skip it silently and never fall back to effGen's BashTool (not a sandbox).
    raise ToolBuildError("tool 'shell' is not available yet: the sandboxed shell is task T4.2")


_REGISTRY: dict[str, Callable[[Path], BaseTool]] = {
    "file_operations": _file_operations,
    "shell": _shell,
}
TOOL_NAMES = tuple(_REGISTRY)


def build_tools(names: Sequence[str], workspace: Path | str) -> list[BaseTool]:
    """Build one fresh tool instance per allowlisted name, confined to `workspace`.

    `[]` builds nothing. Names are checked before the workspace, and nothing is built
    unless every name is valid.
    """
    if isinstance(names, str):
        raise ToolBuildError(f"tool names must be a list of names, not the string {names!r}")
    for name in names:
        if name == DELEGATE_TOOL_NAME:
            raise ToolBuildError(f"{DELEGATE_TOOL_NAME!r} is not built here: the harness adds it to the orchestrator only")
        if name not in _REGISTRY:
            raise ToolBuildError(f"unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)} "
                                 "(web and network tools are never available)")
    if len(set(names)) != len(names):
        raise ToolBuildError(f"duplicate tool names in {list(names)!r}")
    root = resolve_workspace(workspace)
    return [_REGISTRY[name](root) for name in names]
