# Exp 1 at 650M

Repeat of the Exp 1 data-inclusion comparison with `facebook/esm2_t33_650M_UR50D` in place of
ESM-2 8M. Referenced in the paper's Supplementary Materials.

## Commands

The runs use the Exp 1 split; this bundle has no `build_split.py` or `figures.py`.

```
python data/fetch_raw_data.py
python experiments/exp1_data_inclusion/build_split.py
python -m thermality.run_experiment experiments/exp1_data_inclusion_650m
```

## Runs

| config | trains | steps | batch |
|---|---|---|---|
| `configs/pretrain.yaml` | the corpus alone (baseline) | 8500 | 100 |
| `configs/pretrain_with_target_train/800_lr3e-4.yaml` | the corpus plus 800 sequences per target cluster | 8500 | 100 |

Memory use is much higher than in the 8M experiments. Each checkpoint is about 2.6 GB and a
run keeps up to 20 of them, so allow about 100 GB of disk for the two runs.

## Outputs

Each run writes its checkpoints to `checkpoints/exp1_data_inclusion_650m/` and its metrics to
`results/runs/exp1_data_inclusion_650m/<run>.tsv`, with columns
`step, split, cluster, rmse, mae, spearmanr, pearsonr, dispersion_ratio`. The `split` column
is `target` or `control` (one row per evaluation cluster and step), `heldout_val` or
`pretrain`.

This bundle has no `collect_results` or `figures.py` step. To compare with the Supplementary
Materials, average `rmse` over the clusters of the `target` and of the `control` split at each
step, for the baseline run and for the 800-sequence run, as `collect_results` does for the 8M
curves of Exp 1 (`results/exp1_target_curves.tsv`, `results/exp1_control_curves.tsv`).
