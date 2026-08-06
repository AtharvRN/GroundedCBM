from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


QUESTIONS = ["concept_presence", "prediction_relevance", "spatial_usefulness"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SG-CBM human-study CSV exports.")
    parser.add_argument("csv_path")
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def binom_two_sided(k: int, n: int) -> float | None:
    if n <= 0:
        return None

    def prob(i: int) -> float:
        return comb(n, i) * (0.5**n)

    observed = prob(k)
    return min(1.0, sum(prob(i) for i in range(n + 1) if prob(i) <= observed + 1e-15))


def summarize_question(rows: List[Dict[str, str]], question: str) -> Dict[str, object]:
    centered = [int(row[question]) for row in rows if row.get(question, "") != ""]
    sg_scores = [value + 3 for value in centered]
    salf_scores = [3 - value for value in centered]
    sg_wins = sum(1 for value in centered if value > 0)
    salf_wins = sum(1 for value in centered if value < 0)
    ties = sum(1 for value in centered if value == 0)
    non_tie = sg_wins + salf_wins
    return {
        "n": len(centered),
        "mean_sg_centered": mean(centered),
        "mean_score_sg": mean(sg_scores),
        "mean_score_salf": mean(salf_scores),
        "std_score_sg_population": pstdev(sg_scores),
        "std_score_salf_population": pstdev(salf_scores),
        "sg_win": sg_wins,
        "tie": ties,
        "salf_win": salf_wins,
        "sg_win_rate_all": sg_wins / len(centered),
        "tie_rate_all": ties / len(centered),
        "salf_win_rate_all": salf_wins / len(centered),
        "sg_rate_non_tie": sg_wins / non_tie if non_tie else None,
        "sign_test_p_two_sided": binom_two_sided(sg_wins, non_tie),
        "counts_by_centered_score": {str(score): sum(1 for value in centered if value == score) for score in [-2, -1, 0, 1, 2]},
        "counts_by_sg_method_score": {str(score): sum(1 for value in sg_scores if value == score) for score in [1, 2, 3, 4, 5]},
    }


def main() -> None:
    args = parse_args()
    path = Path(args.csv_path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    payload = {
        "csv_path": str(path),
        "participant_ids": sorted({row.get("participant_id", "") for row in rows}),
        "dataset_keys": sorted({row.get("dataset_key", "") for row in rows}),
        "sg_variants": sorted({row.get("sg_variant", "") for row in rows}),
        "salf_variants": sorted({row.get("salf_variant", "") for row in rows}),
        "n_rows": len(rows),
        "score_note": "Method scores follow VLG-CBM style: SG score = SG-centered + 3; SALF score = 3 - SG-centered.",
        "questions": {question: summarize_question(rows, question) for question in QUESTIONS},
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
