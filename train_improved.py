from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


def _load_train_module() -> ModuleType:
    train_path = Path(__file__).with_name("train.py")
    spec = spec_from_file_location("train_notebook_compat", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load training module from {train_path}.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TRAIN = _load_train_module()

CONFIG = _TRAIN.CONFIG
apply_cli_overrides = _TRAIN.apply_cli_overrides
main = _TRAIN.main
parse_args = _TRAIN.parse_args
validate_config = _TRAIN.validate_config

__all__ = [
    "CONFIG",
    "apply_cli_overrides",
    "main",
    "parse_args",
    "validate_config",
]


if __name__ == "__main__":
    main()
