# Data sources

Six third-party ddG datasets and three derived tables, in preprocessed form.

The repository `LICENSE` covers the code only. Each dataset here is distributed under the
terms of its original publication, given below.

## Overview

| file | content | rows | location |
|---|---|---|---|
| `mega_smdi.tsv` | mega-scale folding stability | — | Zenodo (141 MB) |
| `fireprot.tsv` | FireProtDB | 3,247 | repository |
| `tmdbs.tsv` | ThermoMutDB, single substitutions | 3,010 | repository |
| `S2648.tsv` | S2648 | 2,648 | repository |
| `S669.tsv` | S669 | 669 | repository |
| `Ssym.tsv` | Ssym | 342 | repository |
| `mega_S2648_…_cluster.tsv` | MMseqs2 cluster map of the wild types | 1,172 | repository |
| `ALL_SEQS_blastp_named.tsv` | all-versus-all blastp table | 27,594 | repository |
| `wt_embeddings.tsv` | ESM-2 embeddings of 883 wild types | 883 | Zenodo (99 MB) |

The Zenodo files (https://doi.org/10.5281/zenodo.22207909) are downloaded and
checksum-verified by `python data/fetch_raw_data.py`.

## Common schema

All datasets share these conventions:

- `wt_uid` = `hashlib.shake_128(wt_seq.encode()).hexdigest(10)`, a content hash of the
  wild-type sequence and the join key across all files. `mutant_uid` is the same hash of the
  mutant sequence.
- `mutation` uses the sequence numbering of `wt_seq`; `mutation_pdb` (structural datasets
  only) uses the numbering of the deposited structure.
- ddG sign: positive = destabilising.

## Datasets

### `mega_smdi.tsv` — mega-scale folding stability

Tsuboyama, K. et al. *Mega-scale experimental analysis of protein folding stability in biology
and design.* Nature 620, 434–444 (2023). `tsuboyama2023megascale`

- Upstream: https://doi.org/10.5281/zenodo.7992926
- Terms: CC BY 4.0
- Preprocessing: single-mutant, double-mutant and insertion/deletion records concatenated into
  one table; uid columns added; ddG placed on the shared sign convention.
- Used as the pretraining corpus and, in Exp 2.1, as the evaluation set.

### `fireprot.tsv` — FireProtDB

Stourac, J. et al. *FireProtDB: database of manually curated protein stability data.* Nucleic
Acids Research 49, D319–D324 (2021). `stourac2021fireprotdb`

- Upstream: https://loschmidt.chemi.muni.cz/fireprotdb/
- Terms: freely downloadable for academic use; no explicit licence stated.
- Preprocessing: reduced to `wt_seq, mutant_seq, ddG` plus the uid columns. No `mutation_pdb`.

### `tmdbs.tsv` — ThermoMutDB

Xavier, J. S. et al. *ThermoMutDB: a thermodynamic database for missense mutations.* Nucleic
Acids Research 49, D475–D479 (2021). `xavier2021thermomutdb`

- Upstream: https://biosig.lab.uq.edu.au/thermomutdb/
- Terms: free for academic use.
- Preprocessing: single-substitution records only; same schema as FireProtDB plus a `d_ddg`
  uncertainty column.

### `S2648.tsv` — S2648

Dehouck, Y., Kwasigroch, J. M., Gilis, D. & Rooman, M. *PoPMuSiC 2.1: a web server for the
estimation of protein stability changes upon mutation and sequence optimality.* Bioinformatics
27, 1710–1716 (2011). `dehouck2011popmusic21`

- Upstream: supplementary material of the publication.
- Terms: journal supplementary material.
- Preprocessing: full annotation kept (`pdb, chain, pH, T, uniprot, organism, protein, gene`),
  both mutation numberings and the uid columns.

### `S669.tsv` — S669

Pancotti, C. et al. *Predicting protein stability changes upon single-point mutation: a
thorough comparison of the available tools on a new dataset.* Briefings in Bioinformatics 23,
bbab555 (2022). `pancotti2022s669`

- Upstream: supplementary material of the publication; also distributed with DDGun at
  https://folding.biofold.org/ddgun/
- Terms: journal supplementary material.
- Preprocessing: as S2648, with an added `reference` column.
- Used by Exp 2.2 and Exp 2.3 only.

### `Ssym.tsv` — Ssym

Pucci, F., Bernaerts, K. V., Kwasigroch, J. M. & Rooman, M. *Quantification of biases in
predictions of protein stability changes upon mutations.* Bioinformatics 34, 3659–3665 (2018).

- Upstream: supplementary material of the publication.
- Terms: journal supplementary material.
- Preprocessing: as S2648, plus the mutant structure columns (`pdb_mutant`, `chain_mutant`).

## Derived tables

To reproduce the paper, use the tables as provided. The scripts below document how each was
built; a rebuilt table will not be identical to the provided one.

### `mega_S2648_S669_Ssym_p53_M_ptmuld_ptmulnr_fireprot_tmdbs_tmdbm_cluster.tsv` — cluster map

MMseqs2 clustering of the wild-type sequences at 30 % identity and 50 % coverage.

- Columns: `cluster, uid, wt_uid, wt_uid_dataset`
- Script: `data/build_cluster_map.py`
- The map covers 883 wild types. 42 of them come from five datasets that are not
  redistributed here (`p53`, `M` (myoglobin), `ptmuld`, `ptmulnr`, `tmdbm`); the script
  rebuilds the map for the other 841.

### `ALL_SEQS_blastp_named.tsv` — BLAST table

All-versus-all `blastp` over the same wild-type sequences, `-outfmt 6`.

- Used by Exp 2.2: `bitscore_norm = bitscore / mean(self-bitscores)`
- Script: `data/build_blast_table.py`. It runs with BLAST+ defaults and does not reproduce
  the row count of the provided table.

### `wt_embeddings.tsv` — wild-type embeddings

Mean-pooled ESM-2 3B (`esm2_t36_3B_UR50D`) embeddings of the 883 wild-type sequences.

- Used by Exp 2.3 for the `d_esm` axis.
- Script: `data/build_wt_embeddings.py`. It embeds the 841 wild types that have a sequence in
  this directory; values may differ from the published file in the last float digits.
