# Exp 2.2 — Sequence-similarity distance sensitivity

How does the gain from fine-tuning on one wild type transfer to another, as a function of the
sequence similarity between the two? Fig. 3, panel b of the paper.

## The split

`build_split.py` writes `data/splits/exp2_bitscore_norm/`:

- `pretrain.tsv` and `heldout_val.tsv` — the pretraining corpus and its validation set.
- One directory per similarity bin — 12 bins of normalised blastp bitscore between
  `0.099_0.2` and `0.95_1.0` — holding up to five wild-type pairs from the same cluster, each
  wild type with more than 1000 records. Every bin has `wt1_train/`, `wt2_train/`,
  `wt1_test/`, `wt2_test/`, one file per pair.

`bitscore_norm` is `bitscore / mean(self-bitscores)` from `data/sources/ALL_SEQS_blastp_named.tsv`.

## The runs

`configs/pretrain.yaml` pretrains on the corpus (8500 steps, batch 400).
`configs/finetune/<lo>_<hi>_wt{1,2}_train.yaml` — 24 runs, two per bin — fine-tune one model
per pair on one wild type's mutants (1200 steps, batch 400) from the pretraining run's last
checkpoint, and evaluate it on both wild types' test sets.

## Commands

```
python data/fetch_raw_data.py
python experiments/exp2_bitscore_norm/build_split.py
python -m thermality.run_experiment experiments/exp2_bitscore_norm
python -m thermality.collect_results experiments/exp2_bitscore_norm
python experiments/exp2_bitscore_norm/figures.py
```

`collect_results` writes `results/exp2_bitscore_norm.tsv` with columns
`pair, wt1, wt2, bin, direction, bitscore_norm, discrepancy`: two rows per pair, one per
direction. Direction 1 is `RMSE(WT1-trained on WT1-test) − RMSE(WT2-trained on WT1-test)`,
direction 2 the mirror image — the paper's discrepancy.

## What to compare

`figures/exp22.svg` against Fig. 3, panel b. The script prints the Spearman correlation
between `bitscore_norm` and the discrepancy, its p-value and the number of points.
