# AutoResearch Pipeline

This folder stores the lightweight research loop used to improve SG-CBM architecture and training settings.

Primary objective:

- Improve CUB/ImageNet classification under sparse final-layer constraints.
- Main scalar objective is `nec_avg` over requested NEC values.
- Tie-breakers are dense/test accuracy, concept quality, and localization.

Loop:

1. Propose a trial from the search space.
2. Train the CBL/final layer.
3. Run sparse NEC sweep.
4. Optionally run concept accuracy and localization.
5. Record metrics into JSONL memory.
6. Use the memory to choose the next trial.

The pipeline intentionally does not edit model code automatically. Architecture/code changes should be proposed as hypotheses first, then implemented explicitly.

## Commands

Create a trial config and runnable command file:

```bash
python scripts/autoresearch.py propose --space autoresearch/search_spaces/cub_sgcbm_acc_nec.json --memory autoresearch/memory/cub_sgcbm_trials.jsonl
```

Record a finished trial:

```bash
python scripts/autoresearch.py record --memory autoresearch/memory/cub_sgcbm_trials.jsonl --trial_id cub_sgcbm_000001 --run_dir /path/to/run --test_acc 0.7632 --nec_json /path/to/nec_metrics.json
```

Show the leaderboard:

```bash
python scripts/autoresearch.py scoreboard --memory autoresearch/memory/cub_sgcbm_trials.jsonl
```

## Memory Schema

Each JSONL row is one trial:

```json
{
  "trial_id": "cub_sgcbm_000001",
  "created_at": "2026-06-16T00:00:00Z",
  "status": "completed",
  "dataset": "cub",
  "objective": "nec_avg",
  "params": {"loss_global_spatial_align_w": 0.1},
  "run_dir": "/workspace/...",
  "metrics": {"test_acc": 0.7632, "nec_avg": 0.7633}
}
```

