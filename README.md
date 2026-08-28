# Sequence-similarity fine-tuning for ddG prediction

Code and data for the paper *Protein sequence similarity governs interpolation in ΔΔG
prediction*: how the sequence similarity between a fine-tuning set and an evaluation set
governs how much fine-tuning helps a protein stability (ddG) predictor.

- Repository: https://github.com/LigandPro/Sequence-similarity-finetuning-ddG
- Data record: https://doi.org/10.5281/zenodo.22207909 — the two source files too large for
  GitHub. `python data/fetch_raw_data.py` downloads both and verifies their checksums.
- Model: a siamese cross-attention head over ESM-2 8M, trained to regress ddG from a
  (wild type, mutant) sequence pair.
- Four experiments, each a self-contained bundle under `experiments/`, plus the configs of the
  650M repeat of Exp 1.

This repository contains only code and source data. Every number in the paper is regenerated
by training the models: nothing here records what the authors' runs produced. Run the
pipeline and compare your figures with the paper's.

## What is in the repository

```
data/
  fetch_raw_data.py         downloads the two large source files from Zenodo
  build_cluster_map.py      MMseqs2 clustering of the wild types (optional rebuild)
  build_blast_table.py      all-versus-all blastp (optional rebuild)
  build_wt_embeddings.py    mean-pooled ESM-2 embeddings of the wild types (optional rebuild)
  sources/                  5 ddG datasets + cluster map + BLAST table, and SOURCES.md
experiments/<name>/
  README.md                 the question, the split, the commands, what to compare with
  build_split.py            writes data/splits/<name>/ from data/sources/
  configs/                  one YAML per training run
  figures.py                results/<name>*.tsv -> figures/*.svg
experiments/exp1_data_inclusion_650m/   two configs for the 650M repeat of Exp 1
figures/point_cloud.py      the data-partition schematic (Fig. 1, panel c), synthetic
src/thermality/
  config.py                 yacs defaults; a run is identified by its config path
  train.py                  pretraining: one model, evaluated per cluster
  finetune.py               fine-tuning: one model per cluster from a pretrained checkpoint
  run_experiment.py         runs every config of a bundle in order on one GPU
  collect_results.py        results/runs/<name>/*.tsv -> results/<name>*.tsv
  model/                    the architecture, datasets, collation, metrics
```

Everything the pipeline writes is git-ignored: `data/splits/`, `checkpoints/`, `results/`,
`figures/*.svg`, `logs/`.

## Install

```
uv sync
```

Python 3.10 or newer. `uv sync` installs the pinned dependency set from `uv.lock`.

`torch` is declared as `torch>=2.2` and resolves to whatever build your index serves. Training
needs a build matching your CUDA driver; if `torch.cuda.is_available()` is `False` on a machine
with a working GPU, reinstall torch from the wheel index for your driver's CUDA version.
Building splits, collecting results and drawing figures do not need a GPU.

Rebuilding the cluster map needs `mmseqs` on `PATH`; rebuilding the BLAST table needs
`makeblastdb` and `blastp`. Both rebuilds are optional — the tables are committed.

## Hardware

Every experiment runs sequentially on **one GPU**. `thermality.run_experiment` walks a bundle's
configs in order, pretraining first, then the fine-tunes that start from the last checkpoint
the pretraining leaves behind. Set `CUDA_VISIBLE_DEVICES` to pick the GPU. The authors used
A100s; the GPU-hours in the table below are what the fine-tuning stages took on them.

Disk: a pretraining run keeps up to 20 checkpoints of a few tens of MB each; Exp 1 has 11
pretraining runs.

## Reproducing the paper

Six steps per experiment. `<name>` is one of `exp1_data_inclusion`, `exp2_wt_inclusion`,
`exp2_bitscore_norm`, `exp2_d_esm`; each bundle's README gives the exact commands, including
the two `--arm` calls Exp 2.3 needs.

1. **Fetch** the two large source files (once, for all experiments):

   ```
   python data/fetch_raw_data.py
   ```

2. **Rebuild the derived inputs** — optional. The cluster map, the BLAST table and the
   embeddings are committed or fetched; the scripts in `data/` regenerate them and document
   the parameters.

3. **Build the split** for the experiment. Writes `data/splits/<name>/`, deterministically:

   ```
   python experiments/<name>/build_split.py
   ```

4. **Train** every config of the bundle, on one GPU, in order. Each run writes its checkpoints
   under `checkpoints/<name>/` and its per-step, per-cluster metrics to
   `results/runs/<name>/<run>.tsv`:

   ```
   python -m thermality.run_experiment experiments/<name>
   ```

   `--dry-run` prints the commands without running them. A single run is
   `python -m thermality.train -c <config>` or `python -m thermality.finetune -c <config>`.

5. **Collect** the run metrics into the tables the figure reads, `results/<name>*.tsv`:

   ```
   python -m thermality.collect_results experiments/<name>
   ```

6. **Draw the figure** from `results/` alone:

   ```
   python experiments/<name>/figures.py
   ```

   The script writes `figures/*.svg` and prints the summary statistic the paper reports
   (final RMSE, Spearman correlation) for you to compare.

## Experiments

| bundle | question | paper | training | GPU-hours |
|---|---|---|---|---|
| [`exp1_data_inclusion`](experiments/exp1_data_inclusion/README.md) | Does adding target-cluster sequences to pretraining help the target clusters more than the control clusters, and how does that compare with fine-tuning? | Fig. 2; Fig. 1, panel a | 11 pretrains @ 8500 steps, 16 fine-tunes @ 1200 | ~16 for the fine-tunes |
| [`exp2_wt_inclusion`](experiments/exp2_wt_inclusion/README.md) | How much does fine-tuning on a wild type's own mutants help, against fine-tuning on a different wild type's? | Fig. 3, panel a | 1 pretrain @ 8500, 10 fine-tunes @ 1200 | ~59 |
| [`exp2_bitscore_norm`](experiments/exp2_bitscore_norm/README.md) | How does the gain from fine-tuning fall off with the sequence similarity between the two wild types? | Fig. 3, panel b | 1 pretrain @ 8500, 24 fine-tunes @ 1200 | ~79 |
| [`exp2_d_esm`](experiments/exp2_d_esm/README.md) | The same, against ESM embedding distance, within and across clusters | Fig. 3, panel c | 2 pretrains @ 8500, 60 fine-tunes @ 1200 | ~200 |
| [`exp1_data_inclusion_650m`](experiments/exp1_data_inclusion_650m/README.md) | Exp 1 with ESM-2 650M | Supplementary | 2 pretrains @ 8500 | — |

The schematic of Fig. 1, panel c is drawn by `python figures/point_cloud.py` from synthetic
points; it reads no data. The architecture drawing of Fig. 1, panel b is hand-made and has no
generator.

## Data and licence

The code is MIT-licensed; see `LICENSE`. The datasets under `data/sources/` are third-party
redistributions with their own terms, documented per dataset in `data/sources/SOURCES.md`. Two
source files are too large to commit and are published at
https://doi.org/10.5281/zenodo.22207909; `data/fetch_raw_data.py` retrieves them and verifies
their checksums.
