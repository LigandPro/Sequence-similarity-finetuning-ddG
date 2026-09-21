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

Memory use is much higher than in the 8M experiments.
