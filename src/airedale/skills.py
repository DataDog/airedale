# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2026-present Datadog, Inc.

"""Skills staging for provider agent SDKs."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from airedale import js_json
from airedale.pi_frontmatter import read_pi_frontmatter
from airedale.safe_log import exception_summary
from airedale.types import slugify

logger = logging.getLogger(__name__)

# Marker dropped inside each skill we stage so re-staging into the same cwd is
# idempotent while a *repo*-provided skill of the same name (no marker) is
# treated as a collision rather than silently clobbered.
STAGED_MARKER = ".dd-ai-devx-staged"


def stage_skills_for_claude(skill_dirs: list[str], cwd: str) -> list[str]:
    """Stage skill directories for Claude Agent SDK discovery.

    Copies each skill directory to ``<cwd>/.claude/skills/<name>`` and returns
    the staged skill names. Re-staging our own skills is idempotent; a name
    collision with a repo-provided skill raises.

    Args:
        skill_dirs: Paths to skill directories (each containing SKILL.md).
        cwd: Working directory where skills will be staged.

    Returns:
        List of skill names (derived from directory basenames).

    Raises:
        ValueError: If a skill directory doesn't exist or collides with a repo skill.
    """
    names = _stage_skill_dirs(skill_dirs, Path(cwd) / ".claude" / "skills")
    if names:
        logger.info("Staged %d skills for Claude: %s", len(names), ", ".join(names))
    return names


def stage_skills_for_codex(skill_dirs: list[str], cwd: str) -> list[str]:
    """Stage skill directories for Codex discovery.

    Copies each skill directory to ``<cwd>/.codex/skills/<name>`` and returns
    the staged skill names. Re-staging our own skills is idempotent; a name
    collision with a repo-provided skill raises.

    Args:
        skill_dirs: Paths to skill directories (each containing SKILL.md).
        cwd: Working directory where skills will be staged.

    Returns:
        List of skill names (derived from directory basenames).

    Raises:
        ValueError: If a skill directory doesn't exist or collides with a repo skill.
    """
    names = _stage_skill_dirs(skill_dirs, Path(cwd) / ".codex" / "skills")
    if names:
        logger.info("Staged %d skills for Codex: %s", len(names), ", ".join(names))
    return names


def exclude_staged_skills_from_git(cwd: str, skill_names: list[str], *, subdir: str) -> None:
    """Append staged skill paths to the worktree's git exclude file.

    When the run ``cwd`` is a git worktree (e.g. a cloned repo workspace), the
    scenario skills we stage under ``<cwd>/<subdir>/<name>`` would otherwise show
    up as untracked changes to the agent (``git status``). See
    :func:`exclude_paths_from_git`.
    """
    exclude_paths_from_git(cwd, [f"{subdir}/{name}" for name in skill_names])


def exclude_paths_from_git(cwd: str, relative_paths: list[str]) -> None:
    """Add ``relative_paths`` (workspace-relative dirs) to the local git excludes.

    Adds anchored patterns to the repo's local exclude file
    (``$GIT_DIR/info/exclude``, resolved via ``git rev-parse --git-path`` so it
    works for both plain repos and worktrees), hiding harness-created artifacts
    from the agent's untracked-changes view without touching the tracked
    ``.gitignore``. No-op when ``cwd`` is not a git repository, git is
    unavailable, or ``relative_paths`` is empty.
    """
    if not relative_paths:
        return
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--git-path", "info/exclude"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return
    if result.returncode != 0:
        return
    raw = result.stdout.strip()
    if not raw:
        return

    exclude_path = Path(raw)
    if not exclude_path.is_absolute():
        exclude_path = Path(cwd) / exclude_path

    # Anchored directory patterns relative to the worktree root.
    patterns = [f"/{path.strip('/')}/" for path in relative_paths]
    try:
        existing = exclude_path.read_text().splitlines() if exclude_path.exists() else []
        new = [pattern for pattern in patterns if pattern not in existing]
        if not new:
            return
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = "" if not existing or existing[-1] == "" else "\n"
        with exclude_path.open("a", encoding="utf-8") as handle:
            handle.write(prefix + "\n".join(new) + "\n")
    except OSError:
        logger.debug("Unable to update git exclude file at %s", exclude_path, exc_info=True)


def discover_claude_skill_names(cwd: str) -> list[str]:
    """Return all skill names discoverable under ``<cwd>/.claude/skills``.

    This is the scenario ∪ repo skill set (staged scenario skills plus any the
    cloned repo ships), which forms the Claude ``skills`` allow-list.
    """
    return _discover_skill_names(Path(cwd) / ".claude" / "skills")


#: Project skill roots that pi discovers in the working directory, with the
#: discovery mode of each (see ``_pi_skill_entries``).
PI_PROJECT_SKILL_ROOTS: tuple[tuple[str, str], ...] = ((".pi/skills", "pi"), (".agents/skills", "agents"))
#: Directory names that pi does not enter when it looks for skills.
_PI_SKIPPED_SKILL_DIRS = frozenset({"node_modules"})


def pi_skill_dirs(skill_dirs: list[str], cwd: str) -> list[str]:
    """Return the skill paths to pass to pi with ``--skill``, in pi's precedence order.

    The pi harness runs pi with ``--no-skills`` so that user-global skills
    (for example ``~/.agents/skills``) and ``.agents/skills`` above the
    workspace never load. This function returns:

    1. The scenario skill directories.
    2. The paths in the ``skills`` list of the project ``.pi/settings.json``
       (relative to ``.pi``). Override entries (``!``, ``+``, ``-``) are not
       paths and are left out. Pi gives these entries precedence over the
       automatic discovery.
    3. The project root ``.pi/skills``: pi's directory loader then applies
       the same rules as its discovery (recursive, ignore files).
    4. The skill files that pi's discovery finds in ``.agents/skills``. Its
       rules differ from the directory loader (standalone ``.md`` files count
       below the root, not at the root), so the files are passed. Ignore
       files are not applied there.

    Pi keeps the first skill of a name. It names a skill from the ``name`` of
    its frontmatter (else the directory name). Limit: ignore files are not
    applied in the collision check, so a skill that an ignore file excludes
    can still make the run fail (with a clear message; never a silent change). A scenario skill with the name
    of a project skill raises ``ValueError``, as for the other harnesses (pi
    would hide the project skill). Nothing is copied.
    """
    scenario: list[str] = []
    names: dict[str, str] = {}
    real_paths: set[str] = set()  # Pi skips a skill file that it already loaded (a link to it) quietly.
    #: The first alias of every file that was read, also when the skill was shadowed: the names that the links give
    #: to one file are compared before any name is dropped (``_check_same_name_through_aliases``).
    aliases: dict[str, tuple[str, Path]] = {}
    for skill_dir in skill_dirs:
        source_path = Path(skill_dir).resolve()
        if not source_path.exists():
            raise ValueError(f"Skill directory does not exist: {skill_dir}")
        if not source_path.is_dir():
            raise ValueError(f"Skill path is not a directory: {skill_dir}")
        for name, skill_file in _pi_skills(source_path):
            real = os.path.realpath(skill_file)
            _check_aliases(aliases, real, name, skill_file)
            if real in real_paths:
                continue
            if name in names:
                raise ValueError(f"Skill '{name}' (scenario skill {skill_file}) collides with {names[name]}")
            names[name] = f"scenario skill {skill_file}"
            real_paths.add(real)
        scenario.append(str(source_path))

    workspace = Path(cwd)
    candidates = [path for path in _pi_settings_skill_paths(workspace) if path.exists()]
    for subdir, mode in PI_PROJECT_SKILL_ROOTS:
        root = workspace / subdir
        if mode == "pi":
            if root.is_dir():
                candidates.append(root)
        else:
            candidates.extend(_pi_skill_entries(root, mode))
    project: list[str] = []
    seen_project: dict[str, str] = {}
    for path in candidates:
        if str(path) in project:
            continue
        for name, skill_file in _pi_skills(path):
            real = os.path.realpath(skill_file)
            _check_aliases(aliases, real, name, skill_file)
            if real in real_paths:
                # The same file through a symlink: pi loads it once and does not report a collision.
                continue
            if name in names:
                raise ValueError(
                    f"Skill '{name}' (project skill {skill_file}) collides with {names[name]}; "
                    "skill names must be unique"
                )
            if name in seen_project:
                # Pi keeps the first skill of a name.
                logger.warning("Project skill %s shadowed by %s", skill_file, seen_project[name])
                continue
            seen_project[name] = str(skill_file)
            real_paths.add(real)
        project.append(str(path))
    return [*scenario, *project]


def _pi_settings_skill_paths(workspace: Path) -> list[Path]:
    """Return the skill paths of the project ``.pi/settings.json`` (``skills`` list)."""
    settings_file = workspace / ".pi" / "settings.json"
    if not settings_file.is_file():
        return []
    try:
        # As pi: invalid UTF-8 bytes become U+FFFD, and a UTF-8 byte order mark is removed before the JSON is parsed.
        data = js_json.loads(settings_file.read_bytes().decode("utf-8", errors="replace").removeprefix("\ufeff"))
    except (OSError, ValueError) as error:
        logger.warning("Unable to parse project pi settings %s (%s)", settings_file, exception_summary(error))
        return []
    entries = data.get("skills") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return []
    paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip() or entry.startswith(("!", "+", "-")):
            continue
        path = _expand_pi_home(entry.strip())
        paths.append(path if path.is_absolute() else (workspace / ".pi" / path))
    return paths


def _expand_pi_home(entry: str) -> Path:
    """Expand a leading ``~`` as pi does (``normalizePath`` of pi): only ``~`` and ``~/...`` (also ``~\\`` on Windows).

    ``Path.expanduser`` also looks up ``~name`` in the user database, and raises ``RuntimeError`` when it cannot: pi
    keeps ``~name/...`` as an ordinary relative path.
    """
    if entry == "~":
        return Path.home()
    if entry.startswith("~/") or (os.name == "nt" and entry.startswith("~\\")):
        # Node's ``join(home, rest)``: separators are merged and ``.`` / ``..`` are resolved (``~//extra`` is
        # ``<home>/extra``; ``Path.home() / "/extra"`` would be ``/extra``).
        return Path(os.path.normpath(f"{Path.home()}{os.sep}{entry[2:]}"))
    return Path(entry)


def _check_aliases(aliases: dict[str, tuple[str, Path]], real: str, name: str, skill_file: Path) -> None:
    """Record the first alias of ``real`` and compare the later ones with it (see ``_check_same_name_through_aliases``)."""
    if real in aliases:
        _check_same_name_through_aliases(aliases[real], name, skill_file)
    else:
        aliases[real] = (name, skill_file)


def _check_same_name_through_aliases(loaded: tuple[str, Path], name: str, skill_file: Path) -> None:
    """Raise when one skill file, reached through two links, gets two names.

    A skill without ``name`` takes the name of its parent directory. Pi loads the file once, from the alias that it
    reads first, and the order in which the system lists a directory is not the same on every system: with different
    names for the aliases, the skill that pi loads (and the collisions it reports) depend on the system. The run is
    refused. Aliases with the same name are the same skill whichever one pi reads first.
    """
    loaded_name, loaded_file = loaded
    if loaded_name != name:
        raise ValueError(
            f"Skill file {os.path.realpath(skill_file)} is reached through links as '{loaded_name}' "
            f"({loaded_file}) and as '{name}' ({skill_file}). It has no 'name', so pi takes the name of the parent "
            "directory of the link that it reads first, and the order of the directory listing depends on the "
            "system. Give the skill a 'name' or remove one of the links"
        )


def _refuse_link_cycle(directory: Path, root: Path, ancestors: frozenset[str]) -> frozenset[str]:
    """Return ``ancestors`` with ``directory``; raise when ``directory`` is already on the path of the walk.

    A symbolic link that leads back to a directory that is being read is a cycle. Pi follows it until the system
    refuses (the limit of links of the OS), and then takes the skills, and for a skill with no ``name`` the name of
    the parent directory, from the alias that it meets first: this depends on the system and cannot be copied. The run
    is refused before pi starts. The message has the skill root and the link, never the content of a file.
    """
    real = os.path.realpath(directory)
    if real in ancestors:
        raise ValueError(
            f"Skill directory {root} has a symbolic link cycle: {directory} leads back to a directory that is "
            "already being read. Remove the link; pi would follow it until the system refuses, and choose the "
            "skill names from the link that it meets first"
        )
    return ancestors | {real}


def _pi_skill_entries(
    directory: Path, mode: str, root: Path | None = None, ancestors: frozenset[str] = frozenset()
) -> list[Path]:
    """Return the skill files that pi's automatic discovery finds in ``directory``.

    Same walk as pi's ``collectSkillEntries``: a directory with ``SKILL.md``
    gives that file and pi does not go deeper; else pi goes into the
    subdirectories (not ``.*``, not ``node_modules``). Standalone ``.md``
    files count only at the root in ``pi`` mode (``.pi/skills``) and only
    below the root in ``agents`` mode (``.agents/skills``). Directory symlinks
    are followed; one that leads back to a directory of the current path raises
    ``ValueError`` (see ``_refuse_link_cycle``). Not reproduced:
    ``.gitignore``-style ignore files.
    """
    root = root or directory
    if not directory.is_dir():
        return []
    at_root = directory == root
    ancestors = _refuse_link_cycle(directory, root, ancestors)
    if (directory / "SKILL.md").is_file():
        return [directory / "SKILL.md"]
    entries: list[Path] = []
    try:
        children = sorted(directory.iterdir())
    except OSError:
        return entries
    for child in children:
        if child.name.startswith(".") or child.name in _PI_SKIPPED_SKILL_DIRS:
            continue
        if (
            child.is_file()
            and child.suffix == ".md"
            and ((mode == "pi" and at_root) or (mode == "agents" and not at_root))
        ):
            entries.append(child)
        elif child.is_dir():
            entries.extend(_pi_skill_entries(child, mode, root, ancestors))
    return entries


def _pi_skills(path: Path) -> list[tuple[str, Path]]:
    """Return ``(name, file)`` for each skill that pi loads from ``path`` (a ``--skill`` value).

    A file is one skill. For a directory, the same walk as pi's
    ``loadSkillsFromDir``: a directory with ``SKILL.md`` is one skill; else pi
    goes into the subdirectories (not ``.*``, not ``node_modules``) and, at the
    top level only, loads the ``.md`` files. Not reproduced: ignore files and
    the validation that makes pi skip a malformed skill.
    """
    if path.is_file():
        name = _pi_skill_name(path)
        return [(name, path)] if name else []
    return _pi_skills_in_dir(path, top_level=True, root=path)


def _pi_skills_in_dir(
    directory: Path, *, top_level: bool, root: Path, ancestors: frozenset[str] = frozenset()
) -> list[tuple[str, Path]]:
    ancestors = _refuse_link_cycle(directory, root, ancestors)
    skill_file = directory / "SKILL.md"
    if skill_file.is_file():
        name = _pi_skill_name(skill_file)
        return [(name, skill_file)] if name else []
    found: list[tuple[str, Path]] = []
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return found
    for entry in entries:
        if entry.name.startswith(".") or entry.name in _PI_SKIPPED_SKILL_DIRS:
            continue
        if entry.is_dir():
            found.extend(_pi_skills_in_dir(entry, top_level=False, root=root, ancestors=ancestors))
        elif top_level and entry.is_file() and entry.suffix == ".md":
            name = _pi_skill_name(entry)
            if name:
                found.append((name, entry))
    return found


def _pi_skill_name(skill_file: Path) -> str | None:
    """Return the name pi gives the skill in ``skill_file`` (``None``: pi does not load it).

    The frontmatter ``name`` when it is a string, else the name of the parent
    directory. As in pi, a file without a non-blank ``description`` is not a
    skill (also ``SKILL.md``).
    """
    frontmatter = _frontmatter(skill_file)
    if frontmatter is None or not frontmatter.get("description", "").strip():
        return None
    return frontmatter.get("name") or skill_file.parent.name


def _frontmatter(path: Path) -> dict[str, str] | None:
    """Return the string values of the top-level YAML frontmatter of ``path`` (see :func:`read_pi_frontmatter`)."""
    return read_pi_frontmatter(path)


def _discover_skill_names(skills_base: Path) -> list[str]:
    if not skills_base.is_dir():
        return []
    return sorted(d.name for d in skills_base.iterdir() if d.is_dir() and (d / "SKILL.md").is_file())


def _stage_skill_dirs(skill_dirs: list[str], skills_base: Path) -> list[str]:
    if not skill_dirs:
        return []

    skills_base.mkdir(parents=True, exist_ok=True)
    skill_names: list[str] = []
    for skill_dir in skill_dirs:
        source_path = Path(skill_dir).resolve()
        if not source_path.exists():
            raise ValueError(f"Skill directory does not exist: {skill_dir}")
        if not source_path.is_dir():
            raise ValueError(f"Skill path is not a directory: {skill_dir}")

        skill_name = _derive_skill_name(source_path.name)
        skill_names.append(skill_name)

        target_path = skills_base / skill_name
        if target_path.exists():
            if (target_path / STAGED_MARKER).exists():
                logger.debug("Replacing previously staged skill: %s", skill_name)
                shutil.rmtree(target_path)
            else:
                raise ValueError(
                    f"Skill '{skill_name}' (from {source_path}) collides with an existing project "
                    f"skill at {target_path}; scenario and repo skill names must be unique"
                )

        logger.debug("Staging skill %s from %s to %s", skill_name, source_path, target_path)
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        (target_path / STAGED_MARKER).write_text("")

    return skill_names


def _derive_skill_name(directory_name: str) -> str:
    """Derive a skill name from a directory basename.

    Uses slugify to ensure the name is valid for both SDKs.
    """
    return slugify(directory_name)
