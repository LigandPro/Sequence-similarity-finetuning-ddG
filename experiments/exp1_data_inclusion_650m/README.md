# Exp 1 at 650M — configs only

Two configs that repeat the Exp 1 data-inclusion comparison on `facebook/esm2_t33_650M_UR50D`
instead of the 8M checkpoint every other experiment uses: the baseline pretraining and the
800-sequence pretrain-with-target-train arm, both at batch 100 for 8500 steps. The paper's
Supplementary Materials refer to this repeat.

The runs use the Exp 1 split. There is no `build_split.py` or `figures.py` here:

```
python experiments/exp1_data_inclusion/build_split.py
python -m thermality.run_experiment experiments/exp1_data_inclusion_650m
```

Expect a much larger memory footprint than the 8M experiments: 650M parameters at batch 100.
