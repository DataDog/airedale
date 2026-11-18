# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""The private report of a failed run: one file that only the user can read, for raw values that logs must not have.

Pi and Claude runs use it. A raw value of a provider, a gateway, a CLI or a model (stderr, error text) can hold
credentials, so it never goes to an error message, a log line or a span; a failed run writes it here, and the error
or the log line gives the name of the file.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

#: The directory for the private reports of failed runs, when it is not the default.
PI_OUTPUT_DIR_ENV = "AIREDALE_OUTPUT_DIR"


#: Whether ``os.open`` takes ``dir_fd`` on this system (it does not on Windows). Read once, at import.
_OPEN_TAKES_DIR_FD = os.open in os.supports_dir_fd


def open_private_report(name: str) -> tuple[int, Path]:
    """Create the private report ``name`` in the output directory and return its file descriptor and its path.

    The directory is ``$AIREDALE_OUTPUT_DIR`` when it is set, else ``airedale-<uid>`` in the temporary directory of the
    system. Either way it is checked the same way, and without a gap between the check and the use:

    1. the directory is created (mode 0700) if it does not exist;
    2. it is **opened** with ``O_DIRECTORY | O_NOFOLLOW`` (a link in the last place is refused);
    3. owner and mode are checked with ``fstat`` on that descriptor: it must be a directory of the user, with no
       access for the group or others;
    4. the file is created **through that descriptor** (``dir_fd``), with ``O_CREAT | O_EXCL | O_NOFOLLOW`` and mode
       0600: replacing the directory by a link afterwards changes nothing, and an existing file or link is never opened.

    Where ``os.open`` does not take ``dir_fd`` (Windows), no file is written: an ``OSError`` is raised. Raise
    ``OSError`` for every refusal.
    """
    needed = ("O_DIRECTORY", "O_NOFOLLOW")
    if not _OPEN_TAKES_DIR_FD or any(not hasattr(os, flag) for flag in needed):
        raise OSError("this system cannot open the output directory safely (no dir_fd): no report is written")
    configured = os.environ.get(PI_OUTPUT_DIR_ENV)
    uid = os.getuid()
    directory = (
        Path(configured).expanduser() if configured else Path(tempfile.gettempdir()) / f"airedale-{uid}"
    )  # (``expanduser`` raises RuntimeError for an unknown ``~user``: the caller catches it.)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(directory_fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or info.st_mode & 0o077:
            raise OSError(f"{directory} is not a directory of the user that is closed to others (mode 0700)")
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    return fd, directory / name
