# Exp 2.2 — Sequence-similarity distance sensitivity

**Question.** How does the gain from fine-tuning on one wild type transfer to another, as a
function of the sequence similarity between the two?

**Paper.** Fig. 3b.

## Commands

```
python data/fetch_raw_data.py
python experiments/exp2_bitscore_norm/build_split.py
python -m thermality.run_experiment experiments/exp2_bitscore_norm
python -m thermality.collect_results experiments/exp2_bitscore_norm
python experiments/exp2_bitscore_norm/figures.py
```

## Split

`build_split.py` writes `data/splits/exp2_bitscore_norm/`. Wild-type pairs are taken from the
same cluster, each wild type with more than 1000 records, and binned by
`bitscore_norm = bitscore / mean(self-bitscores)`, computed from
`data/sources/ALL_SEQS_blastp_named.tsv`.

| path | content |
|---|---|
| `pretrain.tsv`, `heldout_val.tsv` | pretraining corpus and its validation set |
| `<lo>_<hi>/` | one directory per similarity bin, 12 bins from `0.099_0.2` to `0.95_1.0`, up to five pairs each |
| `<lo>_<hi>/wt{1,2}_train/`, `<lo>_<hi>/wt{1,2}_test/` | training and test mutants of each wild type, one file per pair |

## Runs

| configs | trains | runs | steps | batch |
|---|---|---|---|---|
| `configs/pretrain.yaml` | the corpus | 1 | 8500 | 400 |
| `configs/finetune/<lo>_<hi>_wt{1,2}_train.yaml` | one model per pair on one wild type's mutants, from the last pretraining checkpoint | 24 (two per bin) | 1200 | 400 |

Every fine-tuned model is evaluated on both wild types' test sets.

## Outputs

`collect_results` writes `results/exp2_bitscore_norm.tsv` with columns
`pair, wt1, wt2, bin, direction, bitscore_norm, discrepancy`, two rows per pair:

- direction 1: `RMSE(WT1-trained on WT1-test) − RMSE(WT2-trained on WT1-test)`
- direction 2: the mirror image

This is the discrepancy defined in the paper.

## Compare with the paper

`figures/exp22.svg` against Fig. 3b. The script prints the Spearman correlation between
`bitscore_norm` and the discrepancy, its p-value and the number of points.
