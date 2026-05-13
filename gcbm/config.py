from __future__ import annotations

import json
from typing import Any, Iterable


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def load_flat_config(path: str | None) -> dict[str, Any]:
    """Load the flat JSON/YAML config format used by release scripts."""
    if not path:
        return {}
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if path.endswith((".yaml", ".yml")):
        config: dict[str, Any] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                if ":" not in line:
                    raise SystemExit(f"Unsupported config line in {path!r}: {raw_line.rstrip()!r}")
                key, value = line.split(":", 1)
                config[key.strip()] = parse_scalar(value)
        return config
    raise SystemExit("--config must be a flat JSON/YAML file")


def option_value(argv: list[str], *names: str) -> str | None:
    for idx, token in enumerate(argv):
        if token in names and idx + 1 < len(argv):
            return argv[idx + 1]
        for name in names:
            prefix = name + "="
            if token.startswith(prefix):
                return token[len(prefix):]
    return None


def dataset_from_argv_or_config(argv: list[str]) -> str | None:
    config = load_flat_config(option_value(argv, "--config"))
    dataset = option_value(argv, "--dataset") or config.get("dataset")
    return str(dataset).lower() if dataset else None


def model_from_argv_or_config(argv: list[str], config: dict[str, Any]) -> str:
    model = option_value(argv, "--model", "--model_name") or config.get("model") or config.get("model_name") or "sgcbm"
    return str(model).lower().replace("_", "-")


def append_option(argv: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(name)
        return
    argv.extend([name, str(value)])


def config_to_argv(config: dict[str, Any], excluded_keys: Iterable[str] = ()) -> list[str]:
    excluded = {"dataset", "model", "model_name", "config_json", *excluded_keys}
    forwarded: list[str] = []
    for key, value in config.items():
        if key in excluded:
            continue
        append_option(forwarded, f"--{key}", value)
    return forwarded


def strip_dispatcher_args(argv: list[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    dispatcher_options = {"--config", "--dataset", "--model", "--model_name"}
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in dispatcher_options:
            skip_next = True
            continue
        if any(token.startswith(option + "=") for option in dispatcher_options):
            continue
        stripped.append(token)
    return stripped
