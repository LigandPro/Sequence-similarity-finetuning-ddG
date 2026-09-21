# Exp 2.1 — Wild-type inclusion

**Question.** How much does fine-tuning on a wild type's own mutants help, compared with
fine-tuning on a different wild type's mutants?

**Paper.** Fig. 3a.

## Commands

```
python data/fetch_raw_data.py
python experiments/exp2_wt_inclusion/build_split.py
python -m thermality.run_experiment experiments/exp2_wt_inclusion
python -m thermality.collect_results experiments/exp2_wt_inclusion
python experiments/exp2_wt_inclusion/figures.py
```

## Split

`build_split.py` writes `data/splits/exp2_wt_inclusion/`. There are 13 evaluation clusters,
each a pair of wild types (WT1, WT2) from the same cluster.

| path | content |
|---|---|
| `pretrain.tsv`, `heldout_val.tsv` | pretraining corpus and its validation set |
| `wt1_train/<N>/`, `wt2_train/<N>/` | N mutants of WT1 (WT2), one file per cluster. N = 50, 100, 200, 400, 800 |
| `wt1_test/`, `wt2_test/` | held-out mutants of WT1 and WT2, one file per cluster |

## Runs

| configs | trains | runs | steps | batch |
|---|---|---|---|---|
| `configs/pretrain.yaml` | the corpus | 1 | 8500 | 400 |
| `configs/finetune/wt{1,2}_train_<N>_lr1e-4.yaml` | one model per cluster on the N mutants of one wild type, from the last pretraining checkpoint | 10 | 1200 | 400 |

Every fine-tuned model is evaluated on both test sets. For a `wt1` run the matched test set
is `wt1_test` and the mismatched one `wt2_test`; for a `wt2` run the reverse.

## Outputs

`collect_results` writes `results/exp2_wt_inclusion.tsv` with columns
`n_wt_train, arm, direction, cluster, rmse`: the per-cluster RMSE at the last step of every
run on its matched and mismatched test set. N = 0 is the pretrained model evaluated before
the first fine-tuning step.

| column | values |
|---|---|
| `arm` | `matched` — evaluated on the test set of the wild type the model was trained on; `mismatched` — on the other wild type's test set |
| `direction` | `1` — a `wt1_train` run; `2` — a `wt2_train` run |

## Compare with the paper

`figures/exp21.svg` against Fig. 3a. The script prints the N = 0 baseline and the matched and
mismatched RMSE at N = 800, with their relative change.
