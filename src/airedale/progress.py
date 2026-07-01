"""Live progress display for the evaluation matrix.

``ProgressReporter`` drives a ``rich`` live progress bar when enabled and a
terminal is attached, and degrades to plain append-only lines otherwise (the
``--no-progress`` mode). The total number of matrix cells is known up front so
the bar advances once per completed cell; the experiment layer also passes
:meth:`message` as the harness ``ProgressCallback` to surface non-advancing
status updates emitted mid-run.

Keeping the bar pinned below every other line is the whole point of this module.
``rich``'s ``Live`` region (which ``Progress`` builds on) already renders the bar
at the bottom and scrolls other output above it -- but *only* for output that
flows through the same ``Console``. Two classes of output would otherwise punch
through and corrupt the bar:

* ``print(...)`` and any library that writes to ``sys.stdout`` / ``sys.stderr``
  at call time -- handled by ``Progress(redirect_stdout=True,
  redirect_stderr=True)``, which swaps those streams for proxies that forward to
  the live console while the bar is active.
* ``logging`` handlers (ddtrace, the agent SDKs) that captured a direct reference
  to the real ``stderr`` *before* the bar started -- these bypass the stream
  redirect entirely. We therefore detach the root logger's handlers for the
  lifetime of the bar and route logging through a ``RichHandler`` bound to the
  same console, restoring the originals on :meth:`stop`.

The one case we cannot intercept is a logger with ``propagate=False`` *and* its
own handler bound to the original ``stderr``; that is rare and out of scope.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console
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
        self._console: Console | None = None
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._log_handler: logging.Handler | None = None
        self._saved_handlers: list[logging.Handler] | None = None

    def start(self, total: int) -> None:
        """Begin reporting for ``total`` matrix cells."""
        self.total = max(total, 0)
        self.completed = 0
        if not self.enabled or self.total == 0:
            return
        try:
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
            )

            self._console = Console()
            self._progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self._console,
                transient=False,
                # Forward call-time writes to sys.stdout/sys.stderr through the
                # live console so they scroll above the bar instead of over it.
                redirect_stdout=True,
                redirect_stderr=True,
            )
            self._progress.start()
            self._task_id = self._progress.add_task("evals", total=self.total)
            # Reroute logging through the same console so log records emitted by
            # handlers that captured the real stderr cannot punch through the bar.
            self._install_log_routing()
        except Exception:
            # Fall back to plain mode if rich cannot render (e.g. no TTY).
            self._console = None
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
        self._remove_log_routing()
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None
        self._console = None

    def _install_log_routing(self) -> None:
        """Route root-logger output through the live console for the bar's life.

        Any handler bound to the original ``stderr`` before the bar started would
        write straight past ``rich``'s redirect and corrupt the bar. We swap the
        root logger's handlers for a single ``RichHandler`` on our console, which
        ``rich`` renders above the live region; the originals are restored on
        :meth:`stop`. The root logger's level is left untouched so verbosity is
        unchanged.
        """
        if self._console is None:
            return
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                console=self._console,
                show_path=False,
                rich_tracebacks=False,
                markup=False,
            )
            handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
            root = logging.getLogger()
            self._saved_handlers = list(root.handlers)
            for existing in self._saved_handlers:
                root.removeHandler(existing)
            root.addHandler(handler)
            self._log_handler = handler
        except Exception:
            # Logging rerouting is best-effort; never let it break the run.
            self._restore_handlers()

    def _remove_log_routing(self) -> None:
        """Detach the console log handler and restore the original handlers."""
        root = logging.getLogger()
        if self._log_handler is not None:
            root.removeHandler(self._log_handler)
            self._log_handler = None
        self._restore_handlers()

    def _restore_handlers(self) -> None:
        if self._saved_handlers is None:
            return
        root = logging.getLogger()
        for existing in self._saved_handlers:
            if existing not in root.handlers:
                root.addHandler(existing)
        self._saved_handlers = None

    def _plain(self, label: str) -> None:
        """Emit one append-only progress line (plain / no-rich mode)."""
        width = 28
        total = max(self.total, 1)
        filled = min(width, int(width * self.completed / total))
        bar = "#" * filled + "-" * (width - filled)
        print(f"[{bar}] {self.completed}/{self.total} {_clean(label)}", flush=True)
