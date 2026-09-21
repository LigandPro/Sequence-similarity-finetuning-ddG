# Exp 2.3 — Embedding distance sensitivity

**Question.** The question of Exp 2.2 against ESM embedding distance, in two arms: wild-type
pairs from the same cluster (`within`) and from different clusters (`across`).

**Paper.** Fig. 3c.

## Commands

```
python data/fetch_raw_data.py
python experiments/exp2_d_esm/build_split.py --arm within
python experiments/exp2_d_esm/build_split.py --arm across
python -m thermality.run_experiment experiments/exp2_d_esm
python -m thermality.collect_results experiments/exp2_d_esm
python experiments/exp2_d_esm/figures.py
```

## Split

`build_split.py --arm <arm>` writes `data/splits/exp2_d_esm/<arm>/`. Pairs are binned by
`d_esm`, the squared L2 distance between the two wild types' mean-pooled ESM-2 embeddings
(`data/sources/wt_embeddings.tsv`). Bins are 0.5 wide up to 6.0 and 1.0 wide beyond; bins the
data cannot fill are absent.

| path | content |
|---|---|
| `<arm>/pretrain.tsv`, `<arm>/heldout_val.tsv` | the arm's pretraining corpus and its validation set |
| `<arm>/<lo>_<hi>/` | one directory per distance bin, up to five pairs each |
| `<arm>/<lo>_<hi>/wt{1,2}_train/`, `<arm>/<lo>_<hi>/wt{1,2}_test/` | training and test mutants of each wild type, one file per pair |

## Runs

| configs | trains | runs | steps | batch |
|---|---|---|---|---|
| `configs/<arm>/pretrain.yaml` | the arm's corpus | 2 (one per arm) | 8500 | 400 |
| `configs/<arm>/finetune/<lo>_<hi>_wt{1,2}_train.yaml` | one model per pair on one wild type's mutants, from the arm's last pretraining checkpoint | 60 (two per bin) | 1200 | 400 |

Every fine-tuned model is evaluated on both wild types' test sets.

## Outputs

`collect_results` writes `results/exp2_d_esm.tsv` with columns
`pair, wt1, wt2, arm, bin, direction, d_esm, discrepancy`, two rows per pair, one per
direction. The discrepancy is defined as in Exp 2.2.

## Compare with the paper

`figures/exp23.svg` against Fig. 3c. The script prints the Spearman correlation between
`d_esm` and the discrepancy over both arms, its p-value and the number of points.
