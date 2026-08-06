from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample ImageNet validation rows for the human-study UI.")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--val_root", default="/workspace/imagenet_val_full")
    parser.add_argument("--devkit_dir", default="/workspace/imagenet_eval_devkit")
    parser.add_argument("--class_file", default="/workspace/GroundedCBM/concept_files/imagenet_classes.txt")
    parser.add_argument("--output_stem", required=True)
    parser.add_argument("--output_dir", default="/workspace/GroundedCBM/output/human_study")
    parser.add_argument("--exclude_manifest", default="")
    return parser.parse_args()


def load_existing_classes(path: str) -> set[int]:
    if not path:
        return set()
    classes: set[int] = set()
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            classes.add(int(row["class_id_0based"]))
    return classes


def build_class_index(devkit_dir: Path) -> tuple[Dict[int, Any], Dict[str, int]]:
    meta = loadmat(devkit_dir / "data" / "meta.mat", squeeze_me=True)["synsets"]
    leaf_synsets = [synset for synset in meta if int(synset["num_children"]) == 0]
    if len(leaf_synsets) != 1000:
        raise RuntimeError(f"Expected 1000 leaf synsets, found {len(leaf_synsets)}")
    devkit_to_synset = {int(synset["ILSVRC2012_ID"]): synset for synset in leaf_synsets}
    wnids_sorted = sorted(str(synset["WNID"]) for synset in leaf_synsets)
    wnid_to_class_id = {wnid: idx for idx, wnid in enumerate(wnids_sorted)}
    return devkit_to_synset, wnid_to_class_id


def main() -> None:
    args = parse_args()
    val_root = Path(args.val_root)
    devkit_dir = Path(args.devkit_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [line.strip() for line in Path(args.class_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(classes) != 1000:
        raise RuntimeError(f"Expected 1000 class names, found {len(classes)}")

    excluded_classes = load_existing_classes(args.exclude_manifest)
    if args.n > 1000 - len(excluded_classes):
        raise ValueError(f"Cannot sample {args.n} classes after excluding {len(excluded_classes)} classes")

    devkit_to_synset, wnid_to_class_id = build_class_index(devkit_dir)
    gt_path = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
    gt_labels = [int(line.strip()) for line in gt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(gt_labels) != 50000:
        raise RuntimeError(f"Expected 50000 validation labels, found {len(gt_labels)}")

    by_class: Dict[int, List[tuple[int, int, str, str, str, str]]] = {idx: [] for idx in range(1000)}
    for sample_index, devkit_label in enumerate(gt_labels):
        synset = devkit_to_synset[devkit_label]
        wnid = str(synset["WNID"])
        class_id = wnid_to_class_id[wnid]
        filename = f"ILSVRC2012_val_{sample_index + 1:08d}.JPEG"
        path = val_root / filename
        by_class[class_id].append((sample_index, devkit_label, wnid, str(synset["words"]), filename, str(path)))

    eligible_classes = [idx for idx in range(1000) if idx not in excluded_classes]
    rng = random.Random(args.seed)
    selected_class_ids = rng.sample(eligible_classes, int(args.n))
    rows: List[Dict[str, Any]] = []
    for trial_index, class_id in enumerate(selected_class_ids, start=1):
        sample_index, devkit_label, wnid, meta_words, filename, path = rng.choice(by_class[class_id])
        if not Path(path).exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "trial_index": trial_index,
                "selection_seed": args.seed,
                "sample_index_0based": sample_index,
                "val_index_1based": sample_index + 1,
                "filename": filename,
                "path": path,
                "devkit_label_1based": devkit_label,
                "wnid": wnid,
                "class_id_0based": class_id,
                "class_name": classes[class_id],
                "meta_words": meta_words,
                "sampling_note": (
                    f"One ImageNet validation image sampled from each of {args.n} uniformly sampled "
                    f"ImageNet classes; excluded {len(excluded_classes)} classes from {args.exclude_manifest or 'no manifest'}."
                ),
            }
        )

    csv_path = output_dir / f"{args.output_stem}.csv"
    jsonl_path = output_dir / f"{args.output_stem}.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "jsonl": str(jsonl_path),
                "n": len(rows),
                "unique_classes": len({row["class_id_0based"] for row in rows}),
                "excluded_classes": len(excluded_classes),
                "overlap_with_excluded_classes": len({row["class_id_0based"] for row in rows} & excluded_classes),
                "first_row": rows[0],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
