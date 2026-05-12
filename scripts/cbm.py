#!/usr/bin/env python3
"""Basic unified train/test CLI for the release codebase.

The goal of this file is to give users one stable entrypoint while keeping the
existing training/evaluation implementations untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

MODEL_ALIASES = {
    "vlg": "vlg_cbm",
    "lf": "lf_cbm",
    "salf": "salf_cbm",
    "savlg": "savlg_cbm",
    "sgcbm": "savlg_cbm",
    "sg_cbm": "savlg_cbm",
    "sg-cbm": "savlg_cbm",
    "gcbm": "savlg_cbm",
    "g-cbm": "savlg_cbm",
}
MODEL_CHOICES = ("vlg_cbm", "lf_cbm", "salf_cbm", "savlg_cbm")


def _normalize_model_name(name: str) -> str:
    name = name.lower().replace("-", "_")
    return MODEL_ALIASES.get(name, name)


def _run_script(script: str, argv: list[str]) -> None:
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    inserted = False
    try:
        os.chdir(ROOT)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
            inserted = True
        sys.argv = [script, *argv]
        runpy.run_path(str(ROOT / script), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(str(ROOT))
            except ValueError:
                pass


def _parse_scalar(value: str) -> Any:
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


def _load_flat_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if config_path.suffix == ".json":
        return json.loads(config_path.read_text())
    if config_path.suffix in {".yaml", ".yml"}:
        config: dict[str, Any] = {}
        for raw_line in config_path.read_text().splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise SystemExit(f"Unsupported config line in {path!r}: {raw_line!r}")
            key, value = line.split(":", 1)
            config[key.strip()] = _parse_scalar(value)
        return config
    raise SystemExit("--config must be a flat JSON/YAML file")


def _append_option(argv: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(name)
        return
    argv.extend([name, str(value)])


def _config_to_argv(config: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in config.items():
        _append_option(argv, f"--{key}", value)
    return argv


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Flat JSON/YAML config file.")
    parser.add_argument("--dataset", choices=("cub", "imagenet"), help="Dataset to train on.")
    parser.add_argument(
        "--model",
        choices=(*MODEL_CHOICES, "vlg", "lf", "salf", "savlg", "sgcbm", "sg-cbm", "gcbm", "g-cbm"),
        help="Model variant. ImageNet currently supports only SG-CBM/SAVLG.",
    )
    parser.add_argument("--concept_set", help="Concept vocabulary file for CUB training.")
    parser.add_argument("--concept_file", help="Concept vocabulary file for ImageNet training.")
    parser.add_argument("--annotation_dir", help="Concept annotation directory.")
    parser.add_argument("--save_dir", help="Output directory for trained artifacts.")
    parser.add_argument("--load_dir", help="Optional existing run/checkpoint directory.")
    parser.add_argument("--backbone", help="Backbone name for CUB training.")
    parser.add_argument("--feature_layer", help="Backbone feature layer for CUB training.")
    parser.add_argument("--device", help="Device string, e.g. cuda or cpu.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--num_workers", type=int, help="CUB DataLoader worker count.")
    parser.add_argument("--workers", type=int, help="ImageNet DataLoader worker count.")
    parser.add_argument("--val_split", type=float, help="Train/validation split fraction.")
    parser.add_argument("--cbl_epochs", type=int, help="CUB concept bottleneck epochs.")
    parser.add_argument("--epochs", type=int, help="ImageNet concept bottleneck epochs.")
    parser.add_argument("--cbl_batch_size", type=int, help="CUB concept bottleneck batch size.")
    parser.add_argument("--batch_size", type=int, help="ImageNet batch size.")
    parser.add_argument("--saga_lam", type=float, help="Sparse GLM regularization.")
    parser.add_argument("--saga_n_iters", type=int, help="Sparse GLM iterations.")
    parser.add_argument("--saga_batch_size", type=int, help="Sparse GLM batch size.")
    parser.add_argument("--max_train_images", type=int, help="Optional cap on training images.")
    parser.add_argument("--max_test_images", type=int, help="Optional cap on CUB test images.")
    parser.add_argument("--max_val_images", type=int, help="Optional cap on ImageNet validation images.")
    parser.add_argument("--train_root", help="ImageNet train root.")
    parser.add_argument("--val_root", help="Optional ImageNet val root.")
    parser.add_argument("--precomputed_target_dir", help="ImageNet precomputed target directory.")
    parser.add_argument("--skip_test_eval", action="store_true", help="Skip final CUB test evaluation.")
    parser.add_argument("--train_glm_after_cbl", action="store_true", help="Train ImageNet sparse GLM after CBL.")


def cmd_train(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="scripts/cbm.py train",
        description="Train VLG-CBM, LF-CBM, SALF-CBM, or SG-CBM using the basic release interface.",
        epilog="Unknown arguments are forwarded to the underlying trainer.",
    )
    _add_common_train_args(parser)
    args, passthrough = parser.parse_known_args(argv)
    config = _load_flat_config(args.config)

    dataset = args.dataset or config.get("dataset")
    if not dataset:
        raise SystemExit("Provide --dataset or a config with dataset.")

    model_name = _normalize_model_name(args.model or config.get("model_name", "savlg_cbm"))
    if model_name not in MODEL_CHOICES:
        raise SystemExit(f"Unsupported model: {args.model}")

    if dataset == "imagenet":
        if model_name != "savlg_cbm":
            raise SystemExit("ImageNet training in this release currently supports SG-CBM/SAVLG only.")
        forwarded = _config_to_argv({k: v for k, v in config.items() if k not in {"dataset", "model_name", "config_json"}})
        for name in (
            "train_root",
            "val_root",
            "annotation_dir",
            "precomputed_target_dir",
            "concept_file",
            "save_dir",
            "device",
            "seed",
            "workers",
            "val_split",
            "epochs",
            "batch_size",
            "saga_lam",
            "saga_n_iters",
            "saga_batch_size",
            "max_train_images",
            "max_val_images",
        ):
            _append_option(forwarded, f"--{name}", getattr(args, name))
        if args.train_glm_after_cbl:
            forwarded.append("--train_glm_after_cbl")
        _run_script("train_cbm.py", ["--dataset", "imagenet", "--model", "sgcbm", *forwarded, *passthrough])
        return

    if dataset != "cub":
        raise SystemExit(f"Unsupported dataset for basic training: {dataset}")

    forwarded: list[str] = ["--dataset", dataset]
    train_config = config.get("config_json", args.config)
    if train_config:
        forwarded.extend(["--config", str(train_config)])
    forwarded.extend(["--model_name", model_name])
    for name in (
        "concept_set",
        "annotation_dir",
        "save_dir",
        "load_dir",
        "backbone",
        "feature_layer",
        "device",
        "seed",
        "num_workers",
        "val_split",
        "cbl_epochs",
        "cbl_batch_size",
        "saga_lam",
        "saga_n_iters",
        "saga_batch_size",
        "max_train_images",
        "max_test_images",
    ):
        _append_option(forwarded, f"--{name}", getattr(args, name))
    if args.skip_test_eval:
        forwarded.append("--skip_test_eval")
    _run_script("train_cbm.py", [*forwarded, *passthrough])


def cmd_test(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="scripts/cbm.py test",
        description="Run basic sparse GLM/NEC accuracy evaluation for a trained CBM run.",
        epilog="Unknown arguments are forwarded to sparse_evaluation.py.",
    )
    parser.add_argument("--load_path", required=True, help="Trained run directory.")
    parser.add_argument("--lam", type=float, default=None, help="Maximum sparse GLM regularization.")
    parser.add_argument("--result_file", help="Optional CSV path to append summary results.")
    parser.add_argument("--annotation_dir", help="Annotation directory override for VLG-CBM.")
    parser.add_argument("--n_iters", type=int, help="Sparse evaluation iterations.")
    parser.add_argument("--max_glm_steps", type=int, help="Maximum sparse GLM path steps.")
    parser.add_argument("--cbl_batch_size", type=int, help="CBL batch size override for SG-CBM eval.")
    parser.add_argument("--saga_batch_size", type=int, help="Sparse GLM batch size override.")
    parser.add_argument("--num_workers", type=int, help="DataLoader worker override.")
    parser.add_argument("--max_images", type=int, help="Optional cap on evaluation images.")
    parser.add_argument("--lf_cbm", action="store_true", help="Force LF-CBM for legacy runs without metadata.")
    parser.add_argument("--disable_activation_cache", action="store_true", help="Disable activation cache reuse.")
    args, passthrough = parser.parse_known_args(argv)

    forwarded = ["--load_path", args.load_path]
    for name in (
        "lam",
        "result_file",
        "annotation_dir",
        "n_iters",
        "max_glm_steps",
        "cbl_batch_size",
        "saga_batch_size",
        "num_workers",
        "max_images",
    ):
        _append_option(forwarded, f"--{name}", getattr(args, name))
    if args.lf_cbm:
        forwarded.append("--lf-cbm")
    if args.disable_activation_cache:
        forwarded.append("--disable_activation_cache")
    _run_script("sparse_evaluation.py", [*forwarded, *passthrough])


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified basic training/testing commands for SG-CBM release code.")
    parser.add_argument("command", choices=("train", "test"), help="Command to run.")
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        parser.parse_args(sys.argv[1:])
        return
    command, rest = sys.argv[1], sys.argv[2:]
    if command == "train":
        cmd_train(rest)
    elif command == "test":
        cmd_test(rest)
    else:
        parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    main()
