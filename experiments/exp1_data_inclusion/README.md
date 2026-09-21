# Exp 1 — Data inclusion

**Question.** Does adding target-cluster sequences to the pretraining corpus help the target
clusters more than the control clusters, and how does that compare with fine-tuning on the
same sequences?

**Paper.** Fig. 2 (training curves) and Fig. 1a (three-regime comparison).

## Commands

```
python data/fetch_raw_data.py
python experiments/exp1_data_inclusion/build_split.py
python -m thermality.run_experiment experiments/exp1_data_inclusion
python -m thermality.collect_results experiments/exp1_data_inclusion
python experiments/exp1_data_inclusion/figures.py
```

## Split

`build_split.py` writes `data/splits/exp1_data_inclusion/`:

| path | content |
|---|---|
| `pretrain.tsv` | pretraining corpus, single substitutions only |
| `heldout_val.tsv` | validation set of the pretraining runs |
| `test/target_per_cluster/`, `test/control_per_cluster/` | 14 target and 14 control evaluation clusters, one file each, plus the pooled `all_target.tsv` / `all_control.tsv` |
| `target_train/<N>/` | N sequences per target cluster held out for training, one file per cluster, and `pretrain_with_all_target_train.tsv`, the corpus with those sequences added. N = 10, 20, 50, 100, 200, 300, 400, 600, 800, 1000 |

## Runs

28 configs under `configs/`:

| configs | regime | trains | runs | steps |
|---|---|---|---|---|
| `pretrain.yaml` | baseline | the corpus alone (N = 0) | 1 | 8500 |
| `pretrain_with_target_train/` | pretrain with target-train | the corpus plus N target sequences | 10 | 8500 |
| `target_train_after_pretrain/` | fine-tune from pretrained | one model per target cluster on its N sequences, from the baseline's last checkpoint | 9 | 1200 |
| `target_train_only/` | fine-tune from scratch | the same, from random initialisation | 8 | 300 (1200 for N = 20) |

Batch size and learning rate are set in each config; fine-tuning file names carry the
learning rate.

## Outputs

`collect_results` writes:

| file | columns | content |
|---|---|---|
| `results/exp1_target_curves.tsv`, `results/exp1_control_curves.tsv` | `n_target_train, step, rmse_mean, rmse_se` | mean and standard error of the RMSE over the 14 target (control) clusters, per pretraining run and evaluation step |
| `results/exp1_regimes.tsv` | `regime, n_target_train, cluster, step, rmse` | per-cluster RMSE at the last evaluation step of every run |

## Compare with the paper

| figure | paper | notes |
|---|---|---|
| `figures/exp11_target.svg`, `figures/exp11_control.svg` | Fig. 2 | N = 0, 200, 400, 800; shading is the standard error across clusters |
| `figures/exp12.svg` | Fig. 1a | the script prints the baseline and the minimum mean RMSE of each regime |
