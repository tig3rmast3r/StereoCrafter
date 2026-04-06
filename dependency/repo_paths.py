from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return _REPO_ROOT


def repo_path(*parts: str) -> Path:
    return _REPO_ROOT.joinpath(*parts)


def utilities_path(*parts: str) -> Path:
    return repo_path("Utilities", *parts)


def runners_dir(*, create: bool = False) -> Path:
    path = repo_path("runners")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def configs_dir(*, create: bool = False) -> Path:
    path = repo_path("configs")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir(*, create: bool = False) -> Path:
    path = repo_path("logs")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def runner_path(name: str) -> Path:
    return runners_dir() / name


def config_path(name: str, *, create_dir: bool = False) -> Path:
    return configs_dir(create=create_dir) / name


def log_path(name: str, *, create_dir: bool = False) -> Path:
    return logs_dir(create=create_dir) / name


def resolve_repo_path(raw: PathLike) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_REPO_ROOT / path).resolve()
