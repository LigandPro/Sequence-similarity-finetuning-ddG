# Sequence-similarity fine-tuning for ddG prediction

Code and data for the paper *Protein sequence similarity governs interpolation in ΔΔG
prediction*.

The model is a siamese cross-attention head over ESM-2 8M that regresses ddG from a
(wild type, mutant) sequence pair. The experiments measure how the sequence similarity between
a fine-tuning set and an evaluation set determines the gain from fine-tuning.

- Repository: https://github.com/LigandPro/Sequence-similarity-finetuning-ddG
- Data record: https://doi.org/10.5281/zenodo.22207909

## Requirements

- Python 3.10 or newer, and [uv](https://docs.astral.sh/uv/)
- One CUDA GPU (the authors used A100s)
- Disk space for checkpoints: 25–100 GB per experiment, see [Disk space](#disk-space)
- Optional, only for rebuilding the derived tables: `mmseqs`, `makeblastdb` and `blastp` on
  `PATH`

## Installation

```
git clone https://github.com/LigandPro/Sequence-similarity-finetuning-ddG.git
cd Sequence-similarity-finetuning-ddG
uv sync
source .venv/bin/activate
```

`uv sync` installs the dependency set pinned in `uv.lock`. If `torch.cuda.is_available()`
returns `False`, reinstall `torch` from the wheel index that matches your CUDA driver.

## Data

The source datasets are in `data/sources/`; `data/sources/SOURCES.md` lists their origin,
terms and preprocessing. Two files are hosted on Zenodo and are downloaded and
checksum-verified by:

```
python data/fetch_raw_data.py
```

| file | size | content |
|---|---|---|
| `mega_smdi.tsv` | 141 MB | mega-scale stability dataset |
| `wt_embeddings.tsv` | 99 MB | ESM-2 embeddings of the wild types (Exp 2.3) |

## Experiments

| bundle | paper | question | runs | GPU-hours (A100) |
|---|---|---|---|---|
| [`exp1_data_inclusion`](experiments/exp1_data_inclusion/README.md) | Fig. 1a, Fig. 2 | Does adding target-cluster sequences to pretraining help the target clusters more than the control clusters, and how does that compare with fine-tuning? | 11 pretraining, 17 fine-tuning | ~16 (fine-tuning only) |
| [`exp2_wt_inclusion`](experiments/exp2_wt_inclusion/README.md) | Fig. 3a | How much does fine-tuning on a wild type's own mutants help, compared with a different wild type's? | 1 pretraining, 10 fine-tuning | ~59 |
| [`exp2_bitscore_norm`](experiments/exp2_bitscore_norm/README.md) | Fig. 3b | How does the gain from fine-tuning change with the sequence similarity between two wild types? | 1 pretraining, 24 fine-tuning | ~79 |
| [`exp2_d_esm`](experiments/exp2_d_esm/README.md) | Fig. 3c | The same against ESM embedding distance, within and across clusters | 2 pretraining, 60 fine-tuning | ~200 |
| [`exp1_data_inclusion_650m`](experiments/exp1_data_inclusion_650m/README.md) | Supplementary | Exp 1 with ESM-2 650M | 2 pretraining | — |

Each bundle's README describes its split, its runs, the exact commands and the figure to
compare against.

## Reproducing an experiment

`<name>` is one of the four 8M bundles from the table above. The 650M bundle has its own,
shorter command list in its README.

| step | command | output |
|---|---|---|
| 1. Fetch data (once) | `python data/fetch_raw_data.py` | `data/sources/` |
| 2. Build the split | `python experiments/<name>/build_split.py` | `data/splits/<name>/` |
| 3. Train | `python -m thermality.run_experiment experiments/<name>` | `checkpoints/<name>/`, `results/runs/<name>/*.tsv` |
| 4. Collect metrics | `python -m thermality.collect_results experiments/<name>` | `results/*.tsv` (named in each bundle's README) |
| 5. Draw figures | `python experiments/<name>/figures.py` | `figures/*.svg` |

- Splits are deterministic.
- Step 3 runs every config of the bundle sequentially on one GPU: pretraining first, then the
  fine-tuning runs, which start from the last pretraining checkpoint. Select the GPU with
  `CUDA_VISIBLE_DEVICES`. `--dry-run` prints the commands without running them.
- A single run: `python -m thermality.train -c <config>` (pretraining) or
  `python -m thermality.finetune -c <config>` (fine-tuning).
- Step 5 also prints the summary statistics reported in the paper (final RMSE, Spearman
  correlation).
- Steps 2, 4 and 5 do not need a GPU.

### Disk space

A checkpoint of the 8M model is about 100 MB. A pretraining run keeps up to 20 checkpoints
(about 2 GB); a fine-tuning run keeps up to 2 checkpoints for each model it trains (about
0.2 GB per cluster or wild-type pair). Approximate totals under `checkpoints/`:

| bundle | pretraining | fine-tuning |
|---|---|---|
| `exp1_data_inclusion` | 22 GB (11 runs) | 48 GB (17 runs × 14 clusters) |
| `exp2_wt_inclusion` | 2 GB | 26 GB (10 runs × 13 clusters) |
| `exp2_bitscore_norm` | 2 GB | 23 GB (57 pairs × 2 directions) |
| `exp2_d_esm` | 4 GB (2 runs) | 58 GB (146 pairs × 2 directions) |
| `exp1_data_inclusion_650m` | 100 GB (2 runs, about 2.6 GB per checkpoint) | — |

Steps 4 and 5 read only `results/`, so the fine-tuning checkpoints can be deleted once a
bundle has finished training. Keep the pretraining checkpoints for as long as fine-tuning
runs still have to start from them.

Fig. 1c (the data-partition schematic) is drawn from synthetic points by
`python figures/point_cloud.py`, which writes `figures/data.svg`.

### Rebuilding the derived tables (optional)

The cluster map, the BLAST table and the embeddings are provided. To regenerate them, run the
script with `--out <file>`. Write to a path outside `data/sources/` to keep the provided
tables; `python data/fetch_raw_data.py --force` restores the Zenodo files.

| script | output | needs |
|---|---|---|
| `data/build_cluster_map.py` | MMseqs2 clustering of the wild types | `mmseqs` |
| `data/build_blast_table.py` | all-versus-all blastp table | `makeblastdb`, `blastp` |
| `data/build_wt_embeddings.py` | mean-pooled ESM-2 embeddings of the wild types | GPU recommended |

## Repository layout

```
data/
  fetch_raw_data.py         downloads the Zenodo files
  build_*.py                optional rebuild of the derived tables
  sources/                  source datasets, cluster map, BLAST table, SOURCES.md
experiments/<name>/
  README.md                 split, runs, commands, figure to compare
  build_split.py            data/sources/ -> data/splits/<name>/
  configs/                  one YAML per training run
  figures.py                results/*.tsv -> figures/*.svg
figures/point_cloud.py      Fig. 1c
src/thermality/
  config.py                 config defaults; a run is identified by its config path
  train.py                  pretraining
  finetune.py               fine-tuning, one model per cluster or wild-type pair
  run_experiment.py         runs every config of a bundle
  collect_results.py        per-run metrics -> result tables
  model/                    architecture, datasets, collation, metrics
```

Pipeline outputs (`data/splits/`, `checkpoints/`, `results/`, `figures/*.svg`, `logs/`) are
git-ignored.

## Licence

The code is MIT-licensed (`LICENSE`). The datasets in `data/sources/` are third-party data
under their own terms, listed in `data/sources/SOURCES.md`.
