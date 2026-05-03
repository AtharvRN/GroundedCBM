import datetime
import json
import os
from argparse import Namespace
from typing import Any, Dict


def get_model_name(args: Namespace | Dict[str, Any], default: str = "vlg_cbm") -> str:
    if isinstance(args, dict):
        return str(args.get("model_name", default))
    return str(getattr(args, "model_name", default))


def build_run_dir(base_dir: str, dataset: str, model_name: str) -> str:
    if model_name == "vlg_cbm":
        run_dir = os.path.join(
            base_dir,
            f"{dataset}_cbm_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        )
    else:
        run_dir = os.path.join(
            base_dir,
            f"{model_name}_{dataset}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        )
    while os.path.exists(run_dir):
        run_dir += "-1"
    os.makedirs(run_dir)
    return run_dir


def save_args(args: Namespace, save_dir: str) -> None:
    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        json.dump(vars(args), f, indent=2)


def write_artifacts(save_dir: str, payload: Dict[str, Any]) -> None:
    with open(os.path.join(save_dir, "artifacts.json"), "w") as f:
        json.dump(payload, f, indent=2)


def load_run_info(load_dir: str) -> Dict[str, Any]:
    artifacts_path = os.path.join(load_dir, "artifacts.json")
    args_path = os.path.join(load_dir, "args.txt")
    if os.path.exists(artifacts_path):
        with open(artifacts_path, "r") as f:
            payload = json.load(f)
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                payload.setdefault("args", json.load(f))
        return payload
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            args = json.load(f)
        return {"model_name": get_model_name(args), "args": args}
    raise FileNotFoundError(f"Could not find artifacts.json or args.txt in {load_dir}")

