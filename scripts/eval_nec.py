import argparse
import csv
import json
import math
import runpy
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_nec_values(raw: str) -> set[int] | None:
    if not raw:
        return None
    values = {int(item.strip()) for item in raw.split(",") if item.strip()}
    return values or None


def load_cub_metrics(load_path: Path, nec_values: set[int] | None) -> list[dict[str, Any]]:
    nec_json_path = load_path / "nec_metrics.json"
    if nec_json_path.exists():
        payload = json.loads(nec_json_path.read_text(encoding="utf-8"))
        rows = []
        for row in payload.get("metrics", []):
            nec = int(row["NEC"])
            if nec_values is not None and nec not in nec_values:
                continue
            accuracy = float(row["Accuracy"])
            if math.isfinite(accuracy):
                rows.append({"NEC": nec, "Accuracy": accuracy})
        return rows

    metrics_path = load_path / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} does not exist. Run scripts/train_sparse_nec.py --dataset cub first."
        )
    rows: list[dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            nec_float = float(row["NEC"])
            nec = int(round(nec_float))
            if nec_values is not None and nec not in nec_values:
                continue
            accuracy_raw = (row.get("Accuracy") or "").strip()
            if not accuracy_raw:
                continue
            accuracy = float(accuracy_raw)
            if not math.isfinite(accuracy):
                continue
            rows.append(
                {
                    "NEC": nec,
                    "NEC_raw": nec_float,
                    "Accuracy": accuracy,
                }
            )
    return rows


def load_medical_metrics(path: Path, nec_values: set[int] | None) -> list[dict[str, Any]]:
    sweep_path = path / "glm_nec_sweep_metrics.json"
    if sweep_path.exists():
        payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        rows = []
        for row in payload.get("best_by_nec", []):
            nec = int(row["NEC"])
            if nec_values is not None and nec not in nec_values:
                continue
            metrics = row.get("metrics", {})
            rows.append(
                {
                    "NEC": nec,
                    "nnz": int(row.get("nnz", 0)),
                    "mean_auroc": float(metrics.get("mean_auroc", float("nan"))),
                    "mAP": float(metrics.get("mAP", float("nan"))),
                    "lambda": float(row.get("lambda", float("nan"))),
                    "path_index": int(row.get("path_index", -1)),
                }
            )
        return rows

    nec_path = path / "nec_metrics.json"
    if not nec_path.exists():
        raise FileNotFoundError(f"{path} has neither glm_nec_sweep_metrics.json nor nec_metrics.json")
    payload = json.loads(nec_path.read_text(encoding="utf-8"))
    rows = []
    for row in payload.get("metrics", []):
        nec = int(row["NEC"])
        if nec_values is not None and nec not in nec_values:
            continue
        rows.append(
            {
                "NEC": nec,
                "nnz": int(row.get("nnz", 0)),
                "mean_auroc": float(row.get("mean_auroc", float("nan"))),
                "mAP": float(row.get("mAP", float("nan"))),
            }
        )
    return rows


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate CBM+sparse heads at requested NEC values."
    )
    parser.add_argument("--dataset", required=True, choices=["cub", "imagenet", "chexpert", "mimic", "medical"])
    parser.add_argument("--load_path", default="", help="CUB run directory containing metrics.csv and W_g@NEC checkpoints.")
    parser.add_argument("--artifact_dir", default="", help="ImageNet run or sparse sweep artifact directory.")
    parser.add_argument("--nec_values", default="", help="Comma-separated NEC values to report.")
    parser.add_argument("--output_json", default="", help="Optional output JSON path.")
    return parser.parse_known_args()


def main() -> None:
    args, remaining = parse_args()
    sys.path.insert(0, str(ROOT))
    if args.dataset == "imagenet":
        if not args.artifact_dir:
            raise SystemExit("--artifact_dir is required for --dataset imagenet")
        argv = ["eval_nec.py", "--artifact_dir", args.artifact_dir]
        if args.nec_values:
            argv.extend(["--nec_values", args.nec_values])
        if args.output_json:
            argv.extend(["--output_json", args.output_json])
        sys.argv = [*argv, *remaining]
        runpy.run_path(str(ROOT / "gcbm" / "imagenet_nec.py"), run_name="__main__")
        return

    if args.dataset in {"chexpert", "mimic", "medical"}:
        artifact_path = Path(args.artifact_dir or args.load_path)
        if not str(artifact_path):
            raise SystemExit(f"--load_path or --artifact_dir is required for --dataset {args.dataset}")
        rows = load_medical_metrics(artifact_path, parse_nec_values(args.nec_values))
        payload = {"dataset": args.dataset, "artifact_dir": str(artifact_path), "metrics": rows}
        print(json.dumps(payload, indent=2))
        if args.output_json:
            out = Path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    if not args.load_path:
        raise SystemExit("--load_path is required for --dataset cub")
    rows = load_cub_metrics(Path(args.load_path), parse_nec_values(args.nec_values))
    payload = {"dataset": "cub", "load_path": args.load_path, "metrics": rows}
    print(json.dumps(payload, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
