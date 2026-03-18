"""Unit tests for configs/config_utils.py.

Run with either::

    python -m pytest tests/test_config.py -v
    python -m unittest tests.test_config -v
"""

import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Import target module.  Adjust sys.path so the test runs from the repo root
# whether invoked via pytest or plain unittest.
# ---------------------------------------------------------------------------
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config_utils import (  # noqa: E402
    ConfigNamespace,
    _coerce_value,
    _deep_merge,
    _set_nested,
    get_git_commit_hash,
    load_config,
    load_yaml,
    merge_configs,
    parse_overrides,
    save_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(directory: Path, filename: str, content: dict) -> Path:
    """Write *content* as YAML to *directory/filename* and return the path."""
    path = directory / filename
    with path.open("w") as fh:
        yaml.dump(content, fh, default_flow_style=False, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# ConfigNamespace
# ---------------------------------------------------------------------------


class TestConfigNamespace(unittest.TestCase):
    """Tests for the dot-accessible nested namespace."""

    def _make(self) -> ConfigNamespace:
        return ConfigNamespace(
            {
                "training": {"lr_G": 2e-4, "batch_size": 1},
                "losses": {"lambda_struct": 5.0, "gan_mode": "lsgan"},
                "seed": 42,
            }
        )

    def test_top_level_dot_access(self) -> None:
        cfg = self._make()
        self.assertEqual(cfg.seed, 42)

    def test_nested_dot_access(self) -> None:
        cfg = self._make()
        self.assertAlmostEqual(cfg.training.lr_G, 2e-4)
        self.assertEqual(cfg.training.batch_size, 1)
        self.assertEqual(cfg.losses.gan_mode, "lsgan")

    def test_getitem_access(self) -> None:
        cfg = self._make()
        self.assertEqual(cfg["seed"], 42)
        self.assertEqual(cfg["training"]["batch_size"], 1)

    def test_setitem(self) -> None:
        cfg = self._make()
        cfg["seed"] = 99
        self.assertEqual(cfg.seed, 99)

    def test_setitem_dict_becomes_namespace(self) -> None:
        cfg = ConfigNamespace({})
        cfg["new_section"] = {"key": "value"}
        self.assertIsInstance(cfg.new_section, ConfigNamespace)
        self.assertEqual(cfg.new_section.key, "value")

    def test_contains(self) -> None:
        cfg = self._make()
        self.assertIn("seed", cfg)
        self.assertNotIn("nonexistent", cfg)

    def test_keys(self) -> None:
        cfg = self._make()
        self.assertSetEqual(set(cfg.keys()), {"training", "losses", "seed"})

    def test_to_dict_round_trip(self) -> None:
        original = {
            "training": {"lr_G": 2e-4, "batch_size": 1},
            "losses": {"lambda_struct": 5.0},
            "seed": 42,
        }
        cfg = ConfigNamespace(original)
        self.assertEqual(cfg.to_dict(), original)

    def test_equality(self) -> None:
        d = {"a": {"b": 1}}
        self.assertEqual(ConfigNamespace(d), ConfigNamespace(d))
        self.assertNotEqual(ConfigNamespace({"a": 1}), ConfigNamespace({"a": 2}))

    def test_repr_contains_keys(self) -> None:
        cfg = self._make()
        r = repr(cfg)
        self.assertIn("training", r)
        self.assertIn("seed", r)

    def test_deeply_nested(self) -> None:
        cfg = ConfigNamespace({"a": {"b": {"c": {"d": 42}}}})
        self.assertEqual(cfg.a.b.c.d, 42)


# ---------------------------------------------------------------------------
# _deep_merge / merge_configs
# ---------------------------------------------------------------------------


class TestDeepMerge(unittest.TestCase):
    """Tests for recursive dict merging."""

    def test_non_overlapping_keys_combined(self) -> None:
        result = _deep_merge({"a": 1}, {"b": 2})
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_override_wins_for_scalar(self) -> None:
        result = _deep_merge({"a": 1}, {"a": 99})
        self.assertEqual(result["a"], 99)

    def test_nested_dicts_merged_recursively(self) -> None:
        base = {"training": {"lr_G": 1e-4, "batch_size": 1}}
        override = {"training": {"lr_G": 2e-4}}
        result = _deep_merge(base, override)
        # lr_G updated, batch_size preserved
        self.assertAlmostEqual(result["training"]["lr_G"], 2e-4)
        self.assertEqual(result["training"]["batch_size"], 1)

    def test_base_not_mutated(self) -> None:
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"x": 99}})
        self.assertEqual(base["a"]["x"], 1)

    def test_override_not_mutated(self) -> None:
        override = {"a": {"x": 99}}
        _deep_merge({"a": {"x": 1}}, override)
        self.assertEqual(override["a"]["x"], 99)

    def test_override_scalar_replaces_nested_dict(self) -> None:
        # Override wins even when types differ.
        result = _deep_merge({"a": {"b": 1}}, {"a": 42})
        self.assertEqual(result["a"], 42)

    def test_empty_base(self) -> None:
        result = _deep_merge({}, {"a": 1})
        self.assertEqual(result, {"a": 1})

    def test_empty_override(self) -> None:
        result = _deep_merge({"a": 1}, {})
        self.assertEqual(result, {"a": 1})

    def test_merge_configs_public_alias(self) -> None:
        result = merge_configs({"a": 1}, {"b": 2})
        self.assertEqual(result, {"a": 1, "b": 2})


# ---------------------------------------------------------------------------
# _set_nested
# ---------------------------------------------------------------------------


class TestSetNested(unittest.TestCase):
    """Tests for dot-path nested dict assignment."""

    def test_single_key(self) -> None:
        d: dict = {}
        _set_nested(d, "seed", 42)
        self.assertEqual(d, {"seed": 42})

    def test_two_levels(self) -> None:
        d: dict = {}
        _set_nested(d, "training.batch_size", 4)
        self.assertEqual(d, {"training": {"batch_size": 4}})

    def test_three_levels(self) -> None:
        d: dict = {}
        _set_nested(d, "a.b.c", "hello")
        self.assertEqual(d["a"]["b"]["c"], "hello")

    def test_overwrite_existing(self) -> None:
        d = {"training": {"lr_G": 1e-4}}
        _set_nested(d, "training.lr_G", 9e-4)
        self.assertAlmostEqual(d["training"]["lr_G"], 9e-4)

    def test_intermediate_non_dict_replaced(self) -> None:
        # If an intermediate key holds a scalar, it is replaced with a dict.
        d = {"training": 0}
        _set_nested(d, "training.lr_G", 1e-4)
        self.assertIsInstance(d["training"], dict)


# ---------------------------------------------------------------------------
# _coerce_value
# ---------------------------------------------------------------------------


class TestCoerceValue(unittest.TestCase):
    """Tests for CLI string → Python type coercion."""

    def test_integer(self) -> None:
        self.assertEqual(_coerce_value("4"), 4)
        self.assertIsInstance(_coerce_value("4"), int)

    def test_float(self) -> None:
        self.assertAlmostEqual(_coerce_value("3.14"), 3.14)

    def test_scientific_notation(self) -> None:
        self.assertAlmostEqual(_coerce_value("2.0e-4"), 2e-4)

    def test_bool_true(self) -> None:
        self.assertIs(_coerce_value("true"), True)
        self.assertIs(_coerce_value("True"), True)

    def test_bool_false(self) -> None:
        self.assertIs(_coerce_value("false"), False)
        self.assertIs(_coerce_value("False"), False)

    def test_none_null(self) -> None:
        self.assertIsNone(_coerce_value("null"))
        self.assertIsNone(_coerce_value("~"))

    def test_plain_string(self) -> None:
        self.assertEqual(_coerce_value("lsgan"), "lsgan")

    def test_list(self) -> None:
        self.assertEqual(_coerce_value("[1, 2, 3]"), [1, 2, 3])

    def test_quoted_string_preserved(self) -> None:
        # YAML quoted strings stay as strings.
        self.assertEqual(_coerce_value("'42'"), "42")


# ---------------------------------------------------------------------------
# parse_overrides
# ---------------------------------------------------------------------------


class TestParseOverrides(unittest.TestCase):
    """Tests for CLI override token parsing."""

    def test_equals_syntax(self) -> None:
        result = parse_overrides(["--training.batch_size=4"])
        self.assertEqual(result, {"training": {"batch_size": 4}})

    def test_space_syntax(self) -> None:
        result = parse_overrides(["--training.batch_size", "4"])
        self.assertEqual(result, {"training": {"batch_size": 4}})

    def test_multiple_overrides(self) -> None:
        args = ["--training.batch_size=4", "--losses.lambda_struct=3.0"]
        result = parse_overrides(args)
        self.assertEqual(result["training"]["batch_size"], 4)
        self.assertAlmostEqual(result["losses"]["lambda_struct"], 3.0)

    def test_mixed_syntax(self) -> None:
        args = ["--training.batch_size=4", "--losses.gan_mode", "vanilla"]
        result = parse_overrides(args)
        self.assertEqual(result["training"]["batch_size"], 4)
        self.assertEqual(result["losses"]["gan_mode"], "vanilla")

    def test_non_flag_tokens_skipped(self) -> None:
        # The config path positional arg should be ignored gracefully.
        args = ["configs/experiment_full.yaml", "--training.batch_size=4"]
        result = parse_overrides(args)
        self.assertEqual(result, {"training": {"batch_size": 4}})

    def test_empty_list(self) -> None:
        self.assertEqual(parse_overrides([]), {})

    def test_type_coercion_via_overrides(self) -> None:
        args = [
            "--training.use_amp=true",
            "--training.grad_clip_norm=0.5",
            "--experiment.name=sa_cut_test",
        ]
        result = parse_overrides(args)
        self.assertIs(result["training"]["use_amp"], True)
        self.assertAlmostEqual(result["training"]["grad_clip_norm"], 0.5)
        self.assertEqual(result["experiment"]["name"], "sa_cut_test")

    def test_missing_value_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_overrides(["--training.batch_size"])  # no value following

    def test_missing_value_raises_when_next_is_flag(self) -> None:
        with self.assertRaises(ValueError):
            parse_overrides(["--training.batch_size", "--losses.gan_mode=lsgan"])

    def test_three_level_key(self) -> None:
        result = parse_overrides(["--a.b.c=99"])
        self.assertEqual(result["a"]["b"]["c"], 99)


# ---------------------------------------------------------------------------
# load_yaml
# ---------------------------------------------------------------------------


class TestLoadYaml(unittest.TestCase):
    """Tests for raw YAML file loading."""

    def test_loads_valid_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = _write_yaml(Path(tmpdir), "cfg.yaml", {"seed": 42, "training": {"lr": 1e-3}})
            result = load_yaml(p)
        self.assertEqual(result["seed"], 42)
        self.assertAlmostEqual(result["training"]["lr"], 1e-3)

    def test_empty_yaml_returns_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.yaml"
            path.write_text("")
            result = load_yaml(path)
        self.assertEqual(result, {})

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_yaml("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig(unittest.TestCase):
    """Integration tests for the full load_config pipeline."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

        self.base_dict = {
            "experiment": {"name": "default_run"},
            "training": {"lr_G": 2e-4, "batch_size": 1, "seed": 42},
            "losses": {"lambda_struct": 5.0, "gan_mode": "lsgan"},
        }
        self.exp_dict = {
            "experiment": {"name": "exp_001"},
            "training": {"batch_size": 4},
        }

        self.base_path = _write_yaml(self.root, "default.yaml", self.base_dict)
        self.exp_path = _write_yaml(self.root, "exp.yaml", self.exp_dict)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_experiment_overrides_base(self) -> None:
        cfg = load_config(self.exp_path, base_path=self.base_path, include_git_hash=False)
        self.assertEqual(cfg.experiment.name, "exp_001")

    def test_base_values_preserved_when_not_overridden(self) -> None:
        cfg = load_config(self.exp_path, base_path=self.base_path, include_git_hash=False)
        self.assertAlmostEqual(cfg.training.lr_G, 2e-4)  # from base
        self.assertEqual(cfg.training.batch_size, 4)      # overridden by exp

    def test_no_base_path(self) -> None:
        cfg = load_config(self.exp_path, include_git_hash=False)
        self.assertEqual(cfg.experiment.name, "exp_001")
        # Keys only in base should be absent.
        self.assertNotIn("losses", cfg)

    def test_cli_overrides_applied(self) -> None:
        cli = ["--training.batch_size=8", "--losses.gan_mode=vanilla"]
        cfg = load_config(
            self.exp_path,
            base_path=self.base_path,
            cli_args=cli,
            include_git_hash=False,
        )
        self.assertEqual(cfg.training.batch_size, 8)
        self.assertEqual(cfg.losses.gan_mode, "vanilla")

    def test_cli_overrides_win_over_experiment(self) -> None:
        # exp sets batch_size=4; CLI sets it to 16.
        cli = ["--training.batch_size=16"]
        cfg = load_config(
            self.exp_path,
            base_path=self.base_path,
            cli_args=cli,
            include_git_hash=False,
        )
        self.assertEqual(cfg.training.batch_size, 16)

    def test_meta_injected_when_requested(self) -> None:
        cfg = load_config(self.exp_path, base_path=self.base_path, include_git_hash=True)
        self.assertIn("meta", cfg)
        self.assertIn("config_path", cfg.meta)
        self.assertIn("base_config_path", cfg.meta)
        # git_commit is a string or None — both are acceptable.
        self.assertTrue(
            cfg.meta.git_commit is None or isinstance(cfg.meta.git_commit, str)
        )

    def test_meta_not_injected_when_disabled(self) -> None:
        cfg = load_config(self.exp_path, include_git_hash=False)
        self.assertNotIn("meta", cfg)

    def test_no_cli_args_does_not_read_sys_argv(self) -> None:
        # With cli_args=None (default), no overrides are applied from sys.argv.
        # This verifies scripts importing the module aren't accidentally
        # picking up pytest's own argv tokens.
        cfg = load_config(self.exp_path, base_path=self.base_path, include_git_hash=False)
        # batch_size should come from experiment (4), not anything in sys.argv.
        self.assertEqual(cfg.training.batch_size, 4)


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------


class TestSaveConfig(unittest.TestCase):
    """Tests for YAML serialisation of a resolved config."""

    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = {
                "training": {"lr_G": 2e-4, "batch_size": 1},
                "losses": {"lambda_struct": 5.0},
                "seed": 42,
            }
            cfg = ConfigNamespace(original)
            out_path = root / "resolved.yaml"
            save_config(cfg, out_path)

            with out_path.open("r") as fh:
                reloaded = yaml.safe_load(fh)

        self.assertEqual(reloaded["seed"], 42)
        self.assertAlmostEqual(reloaded["training"]["lr_G"], 2e-4)
        self.assertAlmostEqual(reloaded["losses"]["lambda_struct"], 5.0)

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = Path(tmpdir) / "a" / "b" / "c" / "config.yaml"
            save_config(ConfigNamespace({"x": 1}), deep_path)
            self.assertTrue(deep_path.exists())

    def test_saved_file_reloadable_via_load_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cfg.yaml"
            cfg = ConfigNamespace({"experiment": {"name": "test"}, "seed": 7})
            save_config(cfg, path)
            reloaded = load_yaml(path)
        self.assertEqual(reloaded["experiment"]["name"], "test")
        self.assertEqual(reloaded["seed"], 7)

    def test_full_pipeline_save_and_reload(self) -> None:
        """End-to-end: load_config → save_config → load_yaml → compare."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_path = _write_yaml(root, "default.yaml", {"training": {"lr_G": 2e-4, "seed": 42}})
            exp_path = _write_yaml(root, "exp.yaml", {"training": {"seed": 99}})
            out_path = root / "resolved.yaml"

            cfg = load_config(exp_path, base_path=base_path, include_git_hash=False)
            save_config(cfg, out_path)
            reloaded = load_yaml(out_path)

        self.assertAlmostEqual(reloaded["training"]["lr_G"], 2e-4)
        self.assertEqual(reloaded["training"]["seed"], 99)


# ---------------------------------------------------------------------------
# get_git_commit_hash
# ---------------------------------------------------------------------------


class TestGetGitCommitHash(unittest.TestCase):
    """Tests for git hash retrieval."""

    def test_returns_string_or_none(self) -> None:
        result = get_git_commit_hash()
        self.assertTrue(result is None or isinstance(result, str))

    def test_hash_is_hex_when_present(self) -> None:
        result = get_git_commit_hash()
        if result is not None:
            self.assertEqual(len(result), 40)
            int(result, 16)  # raises ValueError if not hex


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
