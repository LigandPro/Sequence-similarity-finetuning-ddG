# Exp 1 — Data inclusion

Does adding target-cluster sequences to the pretraining corpus help the target clusters more
than the control clusters, and how does that compare with fine-tuning on the same sequences?
One bundle serves two figures of the paper: the training curves (Fig. 2) and the three-regime
comparison (Fig. 1, panel a).

## The split

`build_split.py` writes `data/splits/exp1_data_inclusion/`:

- `pretrain.tsv` — the pretraining corpus, single substitutions only.
- `test/target_per_cluster/`, `test/control_per_cluster/` — 14 target and 14 control
  evaluation clusters, one file each, plus their pooled `all_target.tsv` / `all_control.tsv`.
- `target_train/<N>/` for N in 10, 20, 50, 100, 200, 300, 400, 600, 800, 1000 — N sequences
  per target cluster held out for training, one file per cluster, and
  `pretrain_with_all_target_train.tsv`, the corpus with those sequences added.
- `heldout_val.tsv` — the validation set that selects the pretraining checkpoint.

## The runs

`configs/` holds 28 configs in three groups:

| directory | regime | what it trains |
|---|---|---|
| `pretrain.yaml` | baseline | the corpus alone (N = 0) |
| `pretrain_with_target_train/` | pretrain with target-train | the corpus plus N target sequences, 10 runs |
| `target_train_after_pretrain/` | fine-tune from pretrained | one model per target cluster on its N sequences, starting from the baseline's last checkpoint, 9 runs |
| `target_train_only/` | fine-tune from scratch | the same, from random initialisation, 7 runs |

Pretraining runs 8500 steps at batch 400; fine-tuning 1200 steps at batch 400 (the
from-scratch runs at N ≤ 200 use a lower learning rate, as their file names say).

## Commands

```
python data/fetch_raw_data.py
python experiments/exp1_data_inclusion/build_split.py
python -m thermality.run_experiment experiments/exp1_data_inclusion
python -m thermality.collect_results experiments/exp1_data_inclusion
python experiments/exp1_data_inclusion/figures.py
```

`collect_results` writes three tables:

- `results/exp1_target_curves.tsv`, `results/exp1_control_curves.tsv` — columns
  `n_target_train, step, rmse_mean, rmse_se`: for every pretraining run and evaluation step,
  the mean and standard error of the RMSE over the 14 target (control) clusters.
- `results/exp1_regimes.tsv` — columns `regime, n_target_train, cluster, step, rmse`: the
  per-cluster RMSE at the last evaluation step of every run, under the three regimes above.

## What to compare

- `figures/exp11_target.svg` and `figures/exp11_control.svg` against Fig. 2 of the paper. The
  script plots N = 0, 200, 400, 800 and shades the standard error across clusters; the
  paper's panels show the means alone.
- `figures/exp12.svg` against Fig. 1, panel a. The script prints the baseline and the minimum
  mean RMSE of each regime.
