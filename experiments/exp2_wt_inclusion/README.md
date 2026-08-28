# Exp 2.1 — Wild-type inclusion

How much does fine-tuning on a wild type's own mutants help, compared with fine-tuning on a
different wild type's mutants? Fig. 3, panel a of the paper.

## The split

`build_split.py` writes `data/splits/exp2_wt_inclusion/`:

- `pretrain.tsv` and `heldout_val.tsv` — the pretraining corpus and its validation set.
- 13 evaluation clusters, each a pair of wild types (WT1, WT2) from the same cluster.
- `wt1_train/<N>/`, `wt2_train/<N>/` for N in 50, 100, 200, 400, 800 — N mutants of WT1
  (WT2) per cluster, one file per cluster.
- `wt1_test/`, `wt2_test/` — the held-out mutants of WT1 and WT2 per cluster.

## The runs

`configs/pretrain.yaml` pretrains on the corpus (8500 steps, batch 400).
`configs/finetune/wt{1,2}_train_<N>_lr1e-4.yaml` — 10 runs — each fine-tune one model per
cluster on the N mutants of one wild type, starting from the pretraining run's last
checkpoint, 1200 steps at batch 400, and evaluate it on both wild types' test sets. A `wt1`
run's matched test set is `wt1_test` and its mismatched one `wt2_test`; a `wt2` run the
reverse.

## Commands

```
python data/fetch_raw_data.py
python experiments/exp2_wt_inclusion/build_split.py
python -m thermality.run_experiment experiments/exp2_wt_inclusion
python -m thermality.collect_results experiments/exp2_wt_inclusion
python experiments/exp2_wt_inclusion/figures.py
```

`collect_results` writes `results/exp2_wt_inclusion.tsv` with columns
`n_wt_train, arm, direction, cluster, rmse`: the RMSE at the last step of every run on its
matched and mismatched test set, per cluster, plus the N = 0 point — the pretrained model
evaluated before the first fine-tuning step, which every run logs.

## What to compare

`figures/exp21.svg` against Fig. 3, panel a. The script prints the N = 0 baseline and the
matched and mismatched RMSE at N = 800, with their relative change.
