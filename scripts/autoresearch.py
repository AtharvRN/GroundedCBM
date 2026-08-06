#!/usr/bin/env python3
"""Small experiment planner and memory store for CBM architecture search."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_memory(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_memory(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def params_hash(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def trial_sort_key(trial: dict[str, Any]) -> tuple[float, float]:
    metrics = trial.get("metrics") or {}
    return (float(metrics.get("nec_avg") or -1.0), float(metrics.get("test_acc") or -1.0))


def iter_param_grid(search: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(search)
    values = [search[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def choose_next(space: dict[str, Any], memory: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = dict(space.get("fixed") or {})
    search = space.get("search") or {}
    tried = {params_hash(t.get("params") or {}) for t in memory}

    known_good = space.get("known_good") or []
    if known_good:
        center = dict(known_good[0].get("params") or {})
        params = {**fixed, **center}
        if params_hash(params) not in tried:
            return params

    best = max((t for t in memory if t.get("status") == "completed"), key=trial_sort_key, default=None)
    if best:
        best_params = dict(best.get("params") or {})
        for key, values in search.items():
            for value in values:
                params = {**fixed, **best_params, key: value}
                if params_hash(params) not in tried:
                    return params

    for candidate in iter_param_grid(search):
        params = {**fixed, **candidate}
        if params_hash(params) not in tried:
            return params

    raise SystemExit("No untried configurations remain in this search space.")


def next_trial_id(space_name: str, memory: list[dict[str, Any]]) -> str:
    prefix = space_name.replace("-", "_")
    existing = [t.get("trial_id", "") for t in memory]
    idx = 1
    while f"{prefix}_{idx:06d}" in existing:
        idx += 1
    return f"{prefix}_{idx:06d}"


def write_trial(space_path: Path, memory_path: Path) -> None:
    space = load_json(space_path)
    memory = read_memory(memory_path)
    params = choose_next(space, memory)
    trial_id = next_trial_id(space["name"], memory)

    trial_root = Path(space.get("trial_root", "autoresearch/trials"))
    trial_dir = trial_root / trial_id
    trial_dir.mkdir(parents=True, exist_ok=False)

    base_config = load_json(Path(space["base_config"]))
    config = {**base_config, **params}
    save_root = Path(space.get("save_root", "artifacts/autoresearch")) / trial_id
    config["save_dir"] = str(save_root)

    config_path = trial_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    command_path = trial_dir / "commands.sh"
    command_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"python train_cbm.py --config {config_path}",
                f"RUN_DIR=$(ls -td {save_root}/savlg_cbm_cub_* | head -1)",
                "echo \"RUN_DIR=${RUN_DIR}\"",
                "python scripts/train_sparse_nec.py --dataset cub --load_path \"${RUN_DIR}\" --lam 0.001 --n_iters 4000 --saga_batch_size 512 --cbl_batch_size 32",
                f"python scripts/autoresearch.py record --memory {memory_path} --trial_id {trial_id} --run_dir \"${{RUN_DIR}}\" --nec_json \"${{RUN_DIR}}/nec_metrics.json\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(command_path, 0o755)

    row = {
        "trial_id": trial_id,
        "created_at": utc_now(),
        "status": "proposed",
        "dataset": space.get("dataset"),
        "objective": space.get("objective"),
        "params": params,
        "params_hash": params_hash(params),
        "trial_dir": str(trial_dir),
        "config_path": str(config_path),
        "commands_path": str(command_path),
    }
    append_memory(memory_path, row)
    print(json.dumps(row, indent=2, sort_keys=True))


def load_nec_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        values = [float(r.get("Accuracy") or r.get("accuracy")) for r in rows]
    else:
        payload = load_json(path)
        if isinstance(payload, dict) and "results" in payload:
            results = payload["results"]
        elif isinstance(payload, dict) and "metrics" in payload:
            results = payload["metrics"]
        else:
            results = payload
        if isinstance(results, dict):
            values = []
            for value in results.values():
                if isinstance(value, dict):
                    values.append(float(value.get("accuracy") or value.get("acc") or value.get("top1")))
                else:
                    values.append(float(value))
        else:
            values = [float(item.get("accuracy") or item.get("acc") or item.get("top1")) for item in results]
    return {"nec_avg": sum(values) / len(values)} if values else {}


def first_number(payload: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for value in payload.values():
            found = first_number(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = first_number(value, keys)
            if found is not None:
                return found
    return None


def load_test_metrics(run_dir: str) -> dict[str, float]:
    if not run_dir:
        return {}
    path = Path(run_dir) / "test_metrics.json"
    if not path.exists():
        return {}
    payload = load_json(path)
    value = first_number(payload, ("test_acc", "test_accuracy", "accuracy", "acc"))
    return {"test_acc": value} if value is not None else {}


def load_concept_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = load_json(path)
    results = payload.get("results", payload) if isinstance(payload, dict) else payload
    if isinstance(results, dict) and len(results) == 1:
        results = next(iter(results.values()))
    if not isinstance(results, dict):
        return {}
    metrics = {}
    mapping = {
        "concept_auroc": "auroc",
        "concept_ap": "ap",
        "concept_macro_ap": "macro_ap",
        "concept_p_at_5": "p_at_5",
    }
    for out_key, in_key in mapping.items():
        value = results.get(in_key)
        if isinstance(value, (int, float)):
            metrics[out_key] = float(value)
    best_f1 = results.get("best_f1")
    if isinstance(best_f1, dict) and isinstance(best_f1.get("f1"), (int, float)):
        metrics["best_f1"] = float(best_f1["f1"])
    return metrics


def load_localization_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = load_json(path)
    metrics = {}
    best_mean = payload.get("best_mean_iou") if isinstance(payload, dict) else None
    if isinstance(best_mean, dict) and isinstance(best_mean.get("value"), (int, float)):
        metrics["best_mean_iou"] = float(best_mean["value"])
    best_box = payload.get("best_box_acc") if isinstance(payload, dict) else None
    if isinstance(best_box, dict):
        box_05 = best_box.get("0.5")
        if isinstance(box_05, dict) and isinstance(box_05.get("value"), (int, float)):
            metrics["best_box_acc_at_0p5"] = float(box_05["value"])
    for key in ("soft_iou", "mass_in_gt", "point_hit"):
        value = first_number(payload, (key,))
        if value is not None:
            metrics[key] = value
    return metrics


def record_trial(args: argparse.Namespace) -> None:
    memory_path = Path(args.memory)
    memory = read_memory(memory_path)
    metrics: dict[str, Any] = load_test_metrics(args.run_dir)
    if args.test_acc is not None:
        metrics["test_acc"] = args.test_acc
    if args.nec_json:
        metrics.update(load_nec_metrics(Path(args.nec_json)))
    if args.concept_json:
        metrics.update(load_concept_metrics(Path(args.concept_json)))
    if args.localization_json:
        metrics.update(load_localization_metrics(Path(args.localization_json)))

    updated = False
    for row in memory:
        if row.get("trial_id") == args.trial_id:
            row["status"] = args.status
            row["completed_at"] = utc_now() if args.status == "completed" else row.get("completed_at")
            row["run_dir"] = args.run_dir
            row["metrics"] = {**(row.get("metrics") or {}), **metrics}
            updated = True
            break
    if not updated:
        row = {
            "trial_id": args.trial_id,
            "created_at": utc_now(),
            "status": args.status,
            "run_dir": args.run_dir,
            "metrics": metrics,
        }
        memory.append(row)

    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in memory), encoding="utf-8")
    print(json.dumps(row, indent=2, sort_keys=True))


def print_scoreboard(memory_path: Path) -> None:
    rows = read_memory(memory_path)
    rows = sorted(rows, key=trial_sort_key, reverse=True)
    print("| trial_id | status | nec_avg | test_acc | run_dir |")
    print("|---|---:|---:|---:|---|")
    for row in rows:
        metrics = row.get("metrics") or {}
        nec_avg = metrics.get("nec_avg")
        test_acc = metrics.get("test_acc")
        print(
            "| {trial_id} | {status} | {nec_avg} | {test_acc} | {run_dir} |".format(
                trial_id=row.get("trial_id", ""),
                status=row.get("status", ""),
                nec_avg="" if nec_avg is None else f"{float(nec_avg):.4f}",
                test_acc="" if test_acc is None else f"{float(test_acc):.4f}",
                run_dir=row.get("run_dir", ""),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and record CBM autoresearch experiments.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    propose = sub.add_parser("propose", help="Create the next untried trial.")
    propose.add_argument("--space", required=True)
    propose.add_argument("--memory", required=True)

    record = sub.add_parser("record", help="Record metrics for a trial.")
    record.add_argument("--memory", required=True)
    record.add_argument("--trial_id", required=True)
    record.add_argument("--run_dir", default="")
    record.add_argument("--test_acc", type=float)
    record.add_argument("--nec_json", default="")
    record.add_argument("--concept_json", default="")
    record.add_argument("--localization_json", default="")
    record.add_argument("--status", default="completed", choices=["completed", "failed", "running"])

    scoreboard = sub.add_parser("scoreboard", help="Print trials sorted by objective.")
    scoreboard.add_argument("--memory", required=True)

    args = parser.parse_args()
    if args.cmd == "propose":
        write_trial(Path(args.space), Path(args.memory))
    elif args.cmd == "record":
        record_trial(args)
    elif args.cmd == "scoreboard":
        print_scoreboard(Path(args.memory))


if __name__ == "__main__":
    main()
