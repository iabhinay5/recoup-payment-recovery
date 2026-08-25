"""Where a number came from.

ADR-010 turns a result into a file by recording the configuration beside the metrics. That
only works if every producer of results records the same things the same way, so the
helpers live here rather than in one script that another script later approximates.
"""

from __future__ import annotations

import subprocess

__all__ = ["git_commit", "git_dirty"]


def _git(*args: str) -> str | None:
    """Run a git command, or return None if this is not a usable checkout."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_commit() -> str | None:
    """The commit these numbers were produced under, if this is a checkout."""
    return _git("rev-parse", "HEAD") or None


def git_dirty() -> bool | None:
    """Whether uncommitted changes were present when these numbers were produced.

    Without this the recorded commit promises more than it can keep. ADR-010 offers "here
    is the commit, run it yourself", and from a dirty tree that is not something a reader
    can actually do. A results file produced with local edits is still useful; one that
    does not admit to them is not.
    """
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)
