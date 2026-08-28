"""Regenerate the all-versus-all BLAST table behind Exp 2.2's x-axis.

The committed table is ``data/sources/ALL_SEQS_blastp_named.tsv``. Exp 2.2 bins wild-type pairs
by ``bitscore_norm = bitscore / mean(self-bitscores)``, computed from its ``bitscore`` column;
Exp 2.3 uses the same table only to enumerate candidate pairs.

    python data/build_blast_table.py --out data/sources/regenerated_blastp.tsv

Requires ``makeblastdb`` and ``blastp`` on PATH (NCBI BLAST+).

The filtering cutoffs of the committed table are not recorded: it holds **27,594** pairs
where a full all-versus-all over 883 wild types would produce roughly 1.37 M, so an e-value
threshold or ``-max_target_seqs`` limit was applied. This script uses BLAST+ defaults (e-value
10, ``-max_target_seqs 500``) and will therefore not reproduce the committed table's row count.
The committed table is the one the experiments use; this script documents the computation.

It covers 841 wild types rather than 883: the other 42 have no sequence in ``data/sources/``
and contribute no training row.

The output column list is fixed — the readers index into it by name:

    qseqid sseqid nident pident positive ppos length mismatch gapopen
    qstart qend sstart send evalue bitscore

Sequence ids are ``{wt_uid}_{dataset}``, the same keys as the cluster map. Everything
downstream splits on the underscore and keeps the hash, so a wild type shared by two datasets
appears under both names and is deduplicated on read.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

SOURCES = Path(__file__).resolve().parent / "sources"

COLUMNS = [
    "qseqid", "sseqid", "nident", "pident", "positive", "ppos", "length",
    "mismatch", "gapopen", "qstart", "qend", "sstart", "send", "evalue", "bitscore",
]

# Same tags and files as data/build_cluster_map.py; the two tables must agree on ids.
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
        frame["uid"] = frame["wt_uid"] + "_" + tag
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).drop_duplicates("uid").reset_index(drop=True)


def write_fasta(table: pd.DataFrame, path: Path) -> None:
    with path.open("w") as handle:
        for uid, sequence in zip(table["uid"], table["wt_seq"]):
            handle.write(f">{uid}\n{sequence}\n")


def run_blast(fasta: Path, workdir: Path, threads: int) -> Path:
    for tool in ("makeblastdb", "blastp"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is not on PATH — install NCBI BLAST+ first")

    database = workdir / "wild_types"
    subprocess.run(
        ["makeblastdb", "-in", str(fasta), "-dbtype", "prot", "-out", str(database)],
        check=True,
    )
    hits = workdir / "hits.tsv"
    subprocess.run(
        [
            "blastp",
            "-query", str(fasta),
            "-db", str(database),
            "-out", str(hits),
            "-outfmt", "6 " + " ".join(COLUMNS),
            "-num_threads", str(threads),
        ],
        check=True,
    )
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="destination TSV")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    table = collect_wild_types()
    print(f"{len(table)} sequences into the database")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        fasta = workdir / "wild_types.fasta"
        write_fasta(table, fasta)
        hits = run_blast(fasta, workdir, args.threads)
        frame = pd.read_csv(hits, sep="\t", header=None, names=COLUMNS)

    frame.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out} — {len(frame)} pairs (committed table: 27,594)")


if __name__ == "__main__":
    main()
