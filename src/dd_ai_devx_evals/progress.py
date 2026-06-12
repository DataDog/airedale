"""Live progress display for the evaluation matrix.

``ProgressReporter`` drives a ``rich`` live progress bar when enabled and a
terminal is attached, and degrades to plain append-only lines otherwise (the
``--no-progress`` mode). The total number of matrix cells is known up front so
the bar advances once per completed cell; the experiment layer also passes
:meth:`message` as the harness ``ProgressCallback` to surface non-advancing
status updates emitted mid-run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID


def _clean(label: str, *, limit: int = 120) -> str:
    """Collapse whitespace and truncate a status label for single-line output."""
    cleaned = " ".join(label.split())
    return cleaned[:limit]


class ProgressReporter:
    """Report matrix progress to the console.

    The reporter is safe to use from async code: :meth:`message` is an async
    method (matching the harness ``ProgressCallback`` signature) while
    :meth:`advance`, :meth:`start`, and :meth:`stop` are synchronous and only
    update local display state.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.total = 0
        self.completed = 0
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def start(self, total: int) -> None:
        """Begin reporting for ``total`` matrix cells."""
        self.total = max(total, 0)
        self.completed = 0
        if not self.enabled or self.total == 0:
            return
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
            )

            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                transient=False,
            )
            self._progress.start()
            self._task_id = self._progress.add_task("evals", total=self.total)
        except Exception:
            # Fall back to plain mode if rich cannot render (e.g. no TTY).
            self._progress = None
            self._task_id = None

    def advance(self, label: str) -> None:
        """Mark one matrix cell complete and update the status label."""
        self.completed += 1
        if not self.enabled:
            return
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=1, description=_clean(label))
            return
        self._plain(label)

    async def message(self, label: str) -> None:
        """Render a non-advancing status update (harness ``ProgressCallback``)."""
        if not self.enabled:
            return
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=_clean(label))
            return
        self._plain(label)

    def stop(self) -> None:
        """Stop the live display, flushing any final state."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def _plain(self, label: str) -> None:
        """Emit one append-only progress line (plain / no-rich mode)."""
        width = 28
        total = max(self.total, 1)
        filled = min(width, int(width * self.completed / total))
        bar = "#" * filled + "-" * (width - filled)
        print(f"[{bar}] {self.completed}/{self.total} {_clean(label)}", flush=True)
