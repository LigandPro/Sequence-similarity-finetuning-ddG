# Data sources

Five third-party ddG datasets, one clustering table and one all-versus-all BLAST table are
redistributed here in preprocessed form. Each dataset below names its originating publication,
the upstream location, the terms it is distributed under, and what preprocessing was applied
before it reached this directory.

The repository `LICENSE` covers the **code only**. Nothing in this directory is covered by it.

## Common preprocessing

Every dataset was reduced to a common schema before use. Two conventions are shared:

- **`wt_uid`** is `hashlib.shake_128(wt_seq.encode()).hexdigest(10)` — a content hash of the
  wild-type sequence, and the join key across every file here. `mutant_uid` is the same hash of
  the mutant sequence. A different sequence source therefore produces different ids, different
  clusters and different splits.
- **`mutation` versus `mutation_pdb`.** Structural datasets carry both: `mutation_pdb` uses the
  numbering of the deposited structure, `mutation` uses the sequence numbering of `wt_seq`.
  **Which sequence source and which alignment produced the offset between them is unrecorded**
  and cannot be re-derived from what survives. This is the main reason the preprocessed TSVs are
  committed rather than regenerated from upstream.
- **ddG sign convention.** All files use *positive = destabilising*. The normalisation that
  brought each source onto this convention ran in code that no longer exists.

## Datasets

### `mega_smdi.tsv` — mega-scale folding stability (141 MB, fetched from Zenodo)

Tsuboyama, K. et al. *Mega-scale experimental analysis of protein folding stability in biology
and design.* Nature 620, 434–444 (2023). `tsuboyama2023megascale`

- Upstream: https://doi.org/10.5281/zenodo.7992926
- Terms: CC BY 4.0.
- Preprocessing: single-mutant, double-mutant and insertion/deletion records concatenated into
  one table; `wt_uid` / `mutant_uid` added; ddG placed on the shared sign convention. Provides
  both the pretraining corpus and, in Exp 2.1, the evaluation set.

This file exceeds GitHub's per-file limit and is **not committed**. `data/fetch_raw_data.py`
retrieves it and verifies its checksum.

### `fireprot.tsv` — FireProtDB (3,247 rows)

Stourac, J. et al. *FireProtDB: database of manually curated protein stability data.* Nucleic
Acids Research 49, D319–D324 (2021). `stourac2021fireprotdb`

- Upstream: https://loschmidt.chemi.muni.cz/fireprotdb/
- Terms: freely downloadable for academic use; the project states no explicit license.
- Preprocessing: reduced to `wt_seq, mutant_seq, ddG` plus the uid columns. No structural
  numbering is kept, so there is no `mutation_pdb`.

### `tmdbs.tsv` (3,010 rows) — ThermoMutDB

Xavier, J. S. et al. *ThermoMutDB: a thermodynamic database for missense mutations.* Nucleic
Acids Research 49, D475–D479 (2021). `xavier2021thermomutdb`

- Upstream: https://biosig.lab.uq.edu.au/thermomutdb/
- Terms: free for academic use.
- Preprocessing: split into single-substitution (`tmdbs`) and multi-substitution tables;
  reduced to the same schema as FireProtDB with an extra `d_ddg` uncertainty column. Only the
  single-substitution half is kept — see "Datasets that were dropped" below.

### `S2648.tsv` — S2648 (2,648 rows)

Dehouck, Y., Kwasigroch, J. M., Gilis, D. & Rooman, M. *PoPMuSiC 2.1: a web server for the
estimation of protein stability changes upon mutation and sequence optimality.* Bioinformatics
27, 1710–1716 (2011). `dehouck2011popmusic21`

- Upstream: supplementary material of the publication.
- Terms: journal supplementary material.
- Preprocessing: full annotation kept (`pdb, chain, pH, T, uniprot, organism, protein, gene`)
  plus both mutation numberings and the uid columns.

### `S669.tsv` — S669 (669 rows)

Pancotti, C. et al. *Predicting protein stability changes upon single-point mutation: a
thorough comparison of the available tools on a new dataset.* Briefings in Bioinformatics 23,
bbab555 (2022). `pancotti2022s669`

- Upstream: supplementary material of the publication; also distributed with DDGun at
  https://folding.biofold.org/ddgun/
- Terms: journal supplementary material.
- Preprocessing: as S2648, with an added `reference` column.
- **Read only by Exp 2.2 and Exp 2.3.** The Exp 1 split does not include it, despite the paper
  listing it among the corpus sources.

### `Ssym.tsv` — Ssym (342 rows)

Pucci, F., Bernaerts, K. V., Kwasigroch, J. M. & Rooman, M. *Quantification of biases in
predictions of protein stability changes upon mutations.* Bioinformatics 34, 3659–3665 (2018).

- Upstream: supplementary material of the publication.
- Terms: journal supplementary material.
- Preprocessing: as S2648, keeping the mutant structure columns (`pdb_mutant`, `chain_mutant`)
  that make the set forward/reverse symmetric.

### `mega_S2648_S669_Ssym_p53_M_ptmuld_ptmulnr_fireprot_tmdbs_tmdbm_cluster.tsv` (1,172 rows)

MMseqs2 clustering of all wild-type sequences, at 30 % identity and 50 % coverage. Columns:
`cluster, uid, wt_uid, wt_uid_dataset`. Regenerate with `data/build_cluster_map.py`.

The filename lists the datasets that went in, five of which are no longer part of the corpus —
`p53`, `M` (myoglobin), `ptmuld`, `ptmulnr` and `tmdbm`. Their wild types remain in the map.
The name is kept as-is because every config and split builder refers to the file by it.

Clustering ran over **all** wild types, *before* the single-substitution filter, so the map
contains wild types that no experiment ever trains on.

### `ALL_SEQS_blastp_named.tsv` (27,594 rows)

All-versus-all `blastp` over the same wild-type sequences, in `-outfmt 6` with an explicit
column list. Supplies `bitscore`, from which Exp 2.2 derives its x-axis
`bitscore_norm = bitscore / mean(self-bitscores)`. Regenerate with `data/build_blast_table.py`.

27,594 pairs survive of the ~1.37 M a full all-versus-all would produce, so an e-value or
`-max_target_seqs` cutoff was applied. **The cutoff is unrecorded.**

## Datasets not redistributed

The cluster-map filename lists five datasets that are not in this directory: `p53`,
`Myoglobin`, `ptmuld`, `ptmulnr` and `tmdbm`. They were part of the corpus when the cluster
map and the BLAST table were built, but every experiment filters to single substitutions and
none of their records passes that filter, so they contribute no training and no evaluation
row. 42 of the 883 wild types named in the cluster map and the BLAST table come only from
those files and have no sequence here; the regeneration scripts in `data/` cover the other 841.

### `wt_embeddings.tsv` (883 wild types, fetched from Zenodo)

Not a third-party dataset — mean-pooled ESM-2 embeddings of the wild-type sequences, generated
by `data/build_wt_embeddings.py`. Supplies Exp 2.3's `d_esm` axis. Too large to commit; fetched
alongside `mega_smdi.tsv`.
