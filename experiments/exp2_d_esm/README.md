# Exp 2.3 — Embedding distance sensitivity

The question of Exp 2.2 against embedding distance rather than sequence identity, in two arms:
wild-type pairs from the same cluster and pairs from different clusters. Fig. 3, panel c of
the paper.

## The split

`build_split.py --arm within` and `build_split.py --arm across` write
`data/splits/exp2_d_esm/within/` and `data/splits/exp2_d_esm/across/`. Each arm has its own
`pretrain.tsv` and `heldout_val.tsv`, and one directory per distance bin — bins of the squared
L2 distance `d_esm` between the two wild types' mean-pooled ESM-2 embeddings, 0.5 wide up to
6.0 and 1.0 wide beyond — holding up to five pairs, each with `wt1_train/`, `wt2_train/`,
`wt1_test/`, `wt2_test/`. Bins that the data cannot fill are absent.

The embeddings come from `data/sources/wt_embeddings.tsv`, fetched from Zenodo;
`data/build_wt_embeddings.py` regenerates them.

## The runs

`configs/within/pretrain.yaml` and `configs/across/pretrain.yaml` pretrain one model per arm
(8500 steps, batch 400). `configs/<arm>/finetune/<lo>_<hi>_wt{1,2}_train.yaml` — 60 runs, two
per bin — fine-tune one model per pair on one wild type's mutants (1200 steps, batch 400) from
the arm's pretraining checkpoint, and evaluate it on both wild types' test sets.

## Commands

```
python data/fetch_raw_data.py
python experiments/exp2_d_esm/build_split.py --arm within
python experiments/exp2_d_esm/build_split.py --arm across
python -m thermality.run_experiment experiments/exp2_d_esm
python -m thermality.collect_results experiments/exp2_d_esm
python experiments/exp2_d_esm/figures.py
```

`collect_results` writes `results/exp2_d_esm.tsv` with columns
`pair, wt1, wt2, arm, bin, direction, d_esm, discrepancy`: two rows per pair, one per
direction, with the discrepancy defined as in Exp 2.2.

## What to compare

`figures/exp23.svg` against Fig. 3, panel c. The script prints the Spearman correlation
between `d_esm` and the discrepancy over both arms, its p-value and the number of points.
