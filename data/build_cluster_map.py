"""Regenerate the wild-type cluster map with MMseqs2.

The committed map is
``data/sources/mega_S2648_S669_Ssym_p53_M_ptmuld_ptmulnr_fireprot_tmdbs_tmdbm_cluster.tsv``.
Every split builder reads it: clusters are the unit of the train/test partition in Exp 1 and
the within/across distinction in Exp 2.3.

    python data/build_cluster_map.py --out data/sources/regenerated_cluster.tsv

Requires the ``mmseqs`` binary on PATH (https://github.com/soedinglab/MMseqs2).

Parameters: ``--min-seq-id 0.3 -c 0.5 --cov-mode 1``, as the paper states. The MMseqs2
version the committed map was built with is not recorded, and cluster assignment near the
threshold is version-sensitive, so a rebuilt map need not match the committed one row for row.

Two properties of the map to keep in mind:

1. Clustering runs over **all** wild types, *before* the single-substitution filter.
2. A wild type that appears in several datasets gets one FASTA entry per dataset, keyed
   ``{wt_uid}_{dataset}``. The committed map has 1,172 rows over 883 distinct wild types for
   this reason, and the split builders deduplicate on `wt_uid` when they read it.

The committed map names 42 wild types that have no sequence in ``data/sources/`` (they came
from multi-point datasets that are not redistributed, since no experiment trains on a
multi-point record). This script covers the 841 that do; none of the 42 contributes a training
row, so no split depends on them.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

SOURCES = Path(__file__).resolve().parent / "sources"

MIN_SEQ_ID = 0.3
COVERAGE = 0.5
COV_MODE = 1

# dataset tag -> source file. The tag is what lands in `wt_uid_dataset`.
DATASETS = {
    "mega": "mega_smdi.tsv",
    "S2648": "S2648.tsv",
    "S669": "S669.tsv",
    "Ssym": "Ssym.tsv",
    "fireprot": "fireprot.tsv",
    "tmdbs": "tmdbs.tsv",
}


def collect_wild_types() -> pd.DataFrame:
    rows = []
    for tag, filename in DATASETS.items():
        frame = pd.read_csv(SOURCES / filename, sep="\t", usecols=["wt_uid", "wt_seq"], low_memory=False)
        frame = frame.drop_duplicates("wt_uid")
        frame["wt_uid_dataset"] = tag
        rows.append(frame)
    table = pd.concat(rows, ignore_index=True)
    table["uid"] = table["wt_uid"] + "_" + table["wt_uid_dataset"]
    return table.drop_duplicates("uid").reset_index(drop=True)


def write_fasta(table: pd.DataFrame, path: Path) -> None:
    with path.open("w") as handle:
        for uid, sequence in zip(table["uid"], table["wt_seq"]):
            handle.write(f">{uid}\n{sequence}\n")


def run_mmseqs(fasta: Path, workdir: Path) -> Path:
    if shutil.which("mmseqs") is None:
        raise SystemExit("mmseqs is not on PATH — install MMseqs2 first")
    prefix = workdir / "clusters"
    subprocess.run(
        [
            "mmseqs", "easy-cluster",
            str(fasta), str(prefix), str(workdir / "tmp"),
            "--min-seq-id", str(MIN_SEQ_ID),
            "-c", str(COVERAGE),
            "--cov-mode", str(COV_MODE),
        ],
        check=True,
    )
    return prefix.with_name("clusters_cluster.tsv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="destination TSV")
    args = parser.parse_args()

    table = collect_wild_types()
    print(f"{len(table)} wild-type entries over {table['wt_uid'].nunique()} distinct sequences")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        fasta = workdir / "wild_types.fasta"
        write_fasta(table, fasta)
        result = run_mmseqs(fasta, workdir)
        # MMseqs2 emits `representative<TAB>member`, one row per member.
        pairs = pd.read_csv(result, sep="\t", header=None, names=["cluster", "uid"])

    pairs["wt_uid"] = pairs["uid"].str.rsplit("_", n=1).str[0]
    pairs["wt_uid_dataset"] = pairs["uid"].str.rsplit("_", n=1).str[-1]
    pairs.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out} — {len(pairs)} rows, {pairs['cluster'].nunique()} clusters")


if __name__ == "__main__":
    main()
