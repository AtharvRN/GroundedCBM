# AutoResearch Agent Protocol

Goal: improve classification accuracy under NEC constraints for CBM models.

Primary objective:

- Maximize `nec_avg` across NEC `{5, 10, 15, 20, 25, 30}`.

Secondary objectives:

- Preserve or improve test accuracy.
- Track concept AUROC/AP/P@5.
- Track localization only as a diagnostic, not as the primary optimizer.

Current CUB baseline:

- Best setting: `loss_global_spatial_align_w=0.1`
- Test accuracy: `0.7632`
- NEC average: `0.7633`
- Concept AUROC: `0.9681`
- Concept AP: `0.6448`
- P@5: `0.6535`

Rules:

1. Change one meaningful factor at a time unless running a planned factorial batch.
2. Prefer small moves around the best known setting before broad search.
3. Do not optimize on concept/localization metrics if NEC accuracy regresses materially.
4. Record every completed run into `autoresearch/memory/*.jsonl`.
5. Before proposing code changes, first exhaust config-level hypotheses that are cheap and comparable.
6. For code-level hypotheses, create a short note with:
   - hypothesis
   - expected metric movement
   - exact files/functions touched
   - rollback plan

High-priority next hypotheses:

1. Tune `loss_global_spatial_align_w` around `0.1`: try `0.05` and `0.15`.
2. Tune `loss_mask_w` around `0.25`: try `0.15` and `0.4`.
3. Tune sparse final-layer `saga_lam` around `0.0002`: try `0.0001` and `0.0004`.
4. Revisit global/spatial alignment loss form only after config sweeps saturate.

Expected tool loop:

```bash
python scripts/autoresearch.py scoreboard --memory autoresearch/memory/cub_sgcbm_trials.jsonl
python scripts/autoresearch.py propose --space autoresearch/search_spaces/cub_sgcbm_acc_nec.json --memory autoresearch/memory/cub_sgcbm_trials.jsonl
bash autoresearch/trials/<trial_id>/commands.sh
python scripts/autoresearch.py scoreboard --memory autoresearch/memory/cub_sgcbm_trials.jsonl
```

