"""Configuration loading, merging, and CLI override utilities for SA-CUT.

Typical usage in a training script::

    from configs.config_utils import load_config, save_config

    cfg = load_config(
        config_path="configs/experiment_full.yaml",
        base_path="configs/default.yaml",
        cli_args=sys.argv[1:],   # or pass [] for no CLI overrides
    )
    print(cfg.training.lr_G)    # dot-accessible nested namespace
    save_config(cfg, "results/run_001/resolved_config.yaml")
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Nested namespace
# ---------------------------------------------------------------------------


class ConfigNamespace:
    """A dot-accessible, recursively nested configuration namespace.

    Wraps a plain dict so that ``cfg["training"]["lr_G"]`` can be written
    as ``cfg.training.lr_G``.  Nested dicts are converted automatically.

    Example::

        cfg = ConfigNamespace({"training": {"lr_G": 2e-4}, "seed": 42})
        cfg.training.lr_G   # 0.0002
        cfg.seed            # 42
        cfg.to_dict()       # {"training": {"lr_G": 0.0002}, "seed": 42}
    """

    def __init__(self, d: dict[str, Any]) -> None:
        """Initialise from a (possibly nested) dict.

        Args:
            d: Flat or nested dictionary of configuration values.
        """
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)

    # ------------------------------------------------------------------
    # Dict-style conveniences
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Support ``cfg["key"]`` access."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Support ``cfg["key"] = value`` assignment."""
        if isinstance(value, dict):
            setattr(self, key, ConfigNamespace(value))
        else:
            setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Support ``"key" in cfg``."""
        return hasattr(self, key)

    def keys(self) -> list[str]:
        """Return top-level keys, mirroring dict interface."""
        return list(self.__dict__.keys())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert the namespace back to a plain dict.

        Returns:
            A nested ``dict`` suitable for YAML serialisation.
        """
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            result[key] = value.to_dict() if isinstance(value, ConfigNamespace) else value
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        keys = ", ".join(self.__dict__.keys())
        return f"ConfigNamespace({{{keys}}})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ConfigNamespace):
            return self.to_dict() == other.to_dict()
        return NotImplemented


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Both dicts are left unmodified.  When the same key exists in both and
    both values are dicts, the merge recurses.  Otherwise, the override
    value always wins.

    Args:
        base: Starting configuration (e.g., loaded from ``default.yaml``).
        override: Values that take precedence (e.g., experiment config).

    Returns:
        A new merged dict.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_nested(d: dict[str, Any], key_path: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-delimited key path.

    Creates intermediate dicts as needed.  Modifies *d* in-place.

    Args:
        d: Target dict to modify.
        key_path: Dot-delimited key, e.g. ``"training.batch_size"``.
        value: Value to set at the leaf.

    Example::

        d = {}
        _set_nested(d, "training.batch_size", 4)
        # d == {"training": {"batch_size": 4}}
    """
    keys = key_path.split(".")
    node = d
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _coerce_value(s: str) -> Any:
    """Coerce a CLI string to the most appropriate Python type.

    Uses ``yaml.safe_load`` so that ``"4"`` → ``int``, ``"3.14"`` →
    ``float``, ``"true"`` / ``"false"`` → ``bool``, ``"null"`` → ``None``,
    ``"[1,2,3]"`` → ``list``, and plain strings remain strings.

    Args:
        s: Raw string value from the command line.

    Returns:
        The coerced Python value.
    """
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_git_commit_hash() -> Optional[str]:
    """Return the current HEAD commit hash, or ``None`` if unavailable.

    Works silently when git is not installed or the working directory is
    not inside a git repository.

    Returns:
        A 40-character hex string, or ``None``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed contents as a nested dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as fh:
        data = yaml.safe_load(fh)
    return data if data is not None else {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two config dicts, with *override* taking precedence.

    This is the public wrapper around :func:`_deep_merge`.  Useful when
    you want to compose dicts before constructing a :class:`ConfigNamespace`.

    Args:
        base: Base configuration dict (typically from ``default.yaml``).
        override: Experiment-specific overrides.

    Returns:
        A new merged dict.
    """
    return _deep_merge(base, override)


def parse_overrides(args: list[str]) -> dict[str, Any]:
    """Parse CLI override tokens into a nested dict.

    Recognised formats:

    * ``--training.batch_size=4``   (single token with ``=``)
    * ``--training.batch_size 4``   (two consecutive tokens)

    Tokens that do not start with ``--`` are silently skipped, so it is
    safe to pass the full ``sys.argv[1:]`` list alongside other argparse
    arguments (e.g. ``--config``).

    Values are coerced via :func:`_coerce_value` (YAML semantics), so
    ``"true"`` → ``True``, ``"2.0e-4"`` → ``0.0002``, etc.

    Args:
        args: List of raw command-line tokens.

    Returns:
        A nested dict of overrides ready to deep-merge into a config.

    Raises:
        ValueError: If a ``--key`` token is not followed by a value.

    Example::

        parse_overrides(["--training.batch_size=4", "--losses.lambda_struct=3.0"])
        # {"training": {"batch_size": 4}, "losses": {"lambda_struct": 3.0}}
    """
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            i += 1
            continue

        stripped = token[2:]  # remove leading --

        if "=" in stripped:
            key, value_str = stripped.split("=", maxsplit=1)
        else:
            # Expect the next token to be the value.
            key = stripped
            i += 1
            if i >= len(args) or args[i].startswith("--"):
                raise ValueError(
                    f"CLI override '--{key}' has no value. "
                    f"Use '--{key}=<value>' or '--{key} <value>'."
                )
            value_str = args[i]

        _set_nested(overrides, key, _coerce_value(value_str))
        i += 1

    return overrides


def load_config(
    config_path: str | Path,
    base_path: Optional[str | Path] = None,
    cli_args: Optional[list[str]] = None,
    *,
    include_git_hash: bool = True,
) -> ConfigNamespace:
    """Load, merge, and override a YAML configuration.

    Resolution order (later entries win):

    1. ``base_path`` YAML (e.g. ``configs/default.yaml``) — if provided.
    2. ``config_path`` YAML (experiment-specific overrides).
    3. ``cli_args`` key=value overrides.

    Metadata (git commit hash, resolved file paths) is injected under the
    ``meta`` key when ``include_git_hash=True``.

    Args:
        config_path: Path to the primary (experiment) YAML config file.
        base_path: Optional base config to merge from first. Typically
            ``"configs/default.yaml"``.
        cli_args: List of CLI override tokens such as
            ``["--training.batch_size=4"]``.  Pass ``[]`` to disable CLI
            parsing.  Defaults to ``None``, which means no overrides are
            applied (use :func:`parse_overrides` with ``sys.argv[1:]``
            explicitly in scripts to avoid implicit side-effects).
        include_git_hash: Whether to record the git commit hash and config
            paths under ``cfg.meta``.  Defaults to ``True``.

    Returns:
        A :class:`ConfigNamespace` with dot-accessible nested attributes.

    Raises:
        FileNotFoundError: If *config_path* or *base_path* do not exist.
    """
    config_path = Path(config_path)

    # 1. Base config
    merged: dict[str, Any] = load_yaml(base_path) if base_path is not None else {}

    # 2. Experiment config
    experiment_dict = load_yaml(config_path)
    merged = _deep_merge(merged, experiment_dict)

    # 3. CLI overrides
    if cli_args:
        override_dict = parse_overrides(cli_args)
        merged = _deep_merge(merged, override_dict)

    # 4. Metadata
    if include_git_hash:
        meta: dict[str, Any] = merged.setdefault("meta", {})
        meta["git_commit"] = get_git_commit_hash()
        meta["config_path"] = str(config_path.resolve())
        if base_path is not None:
            meta["base_config_path"] = str(Path(base_path).resolve())

    return ConfigNamespace(merged)


def save_config(cfg: ConfigNamespace, path: str | Path) -> None:
    """Dump a resolved :class:`ConfigNamespace` to a YAML file.

    Creates parent directories if they do not exist.  The saved file can
    be reloaded with :func:`load_config` for exact reproducibility.

    Args:
        cfg: The configuration namespace to serialise.
        path: Destination file path (e.g. ``results/run_001/config.yaml``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.dump(
            cfg.to_dict(),
            fh,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
