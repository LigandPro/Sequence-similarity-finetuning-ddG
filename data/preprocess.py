"""Turn the raw sources into the two tables every split builder reads.

    python data/preprocess.py

Writes ``data/processed/`` (gitignored):

``mutations.tsv``
    Single substitutions from the six sources (mega_smdi, S2648, fireprot, Ssym, tmdbs, S669),
    in that order; a (mutant_seq, wt_uid) that appears in several sources keeps the first.
    Columns: wt_uid, wt_seq, mutant_seq, ddG, cluster, source.
``wt_pairs.tsv``
    One row per unordered wild-type pair (self-pairs kept) with its normalised BLAST bitscore,
    the squared L2 distance between the two ESM embeddings, and whether the wild types share
    a cluster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "sources"
OUT = REPO_ROOT / "data" / "processed"

SOURCES = ["mega_smdi", "S2648", "fireprot", "Ssym", "tmdbs", "S669"]
CLUSTER_FILE = "mega_S2648_S669_Ssym_p53_M_ptmuld_ptmulnr_fireprot_tmdbs_tmdbm_cluster.tsv"


def keep_single_substitutions(dataset: pd.DataFrame) -> pd.DataFrame:
    """Rows whose mutant differs from the wild type at exactly one position."""
    same_length = dataset["wt_seq"].astype(str).str.len() == dataset["mutant_seq"].astype(str).str.len()
    filtered = dataset[same_length]
    differences = np.array(
        [
            sum(a != b for a, b in zip(wt, mutant))
            for wt, mutant in zip(
                filtered["wt_seq"].astype(str).values, filtered["mutant_seq"].astype(str).values
            )
        ]
    )
    return filtered.iloc[differences == 1][["wt_uid", "ddG", "mutant_seq", "wt_seq"]]


def read_clusters() -> pd.DataFrame:
    return pd.read_csv(RAW / CLUSTER_FILE, sep="\t").drop_duplicates(subset=["wt_uid"], keep="first")


def build_mutations() -> pd.DataFrame:
    frames = []
    for source in SOURCES:
        frame = pd.read_csv(RAW / f"{source}.tsv", sep="\t", low_memory=False)
        single = keep_single_substitutions(frame)
        frames.append(single.assign(source=source))
        print(f"{source}: {len(frame)} rows read, {len(frame) - len(single)} not single")

    dataset = pd.concat(frames)
    kept = dataset.drop_duplicates(subset=["mutant_seq", "wt_uid"])
    for source in SOURCES:
        n_single = int((dataset["source"] == source).sum())
        n_kept = int((kept["source"] == source).sum())
        print(f"{source}: {n_single - n_kept} dropped as duplicate, {n_kept} kept")

    kept = kept.merge(read_clusters()[["wt_uid", "cluster"]], on="wt_uid", how="left").reset_index(drop=True)
    kept = kept[["wt_uid", "wt_seq", "mutant_seq", "ddG", "cluster", "source"]]

    assert not kept.duplicated(subset=["mutant_seq", "wt_uid"]).any()
    assert (kept["wt_seq"].str.len() == kept["mutant_seq"].str.len()).all()
    assert all(sum(a != b for a, b in zip(wt, mut)) == 1 for wt, mut in zip(kept["wt_seq"], kept["mutant_seq"]))
    print(f"mutations: {len(kept)} rows")
    return kept


def desymmetrise(blast: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per unordered pair, retaining every self-pair."""
    qseqid = blast["qseqid"].values
    sseqid = blast["sseqid"].values
    pair_df = blast.copy()
    pair_df["dup_key"] = list(zip(np.minimum(qseqid, sseqid), np.maximum(qseqid, sseqid)))
    duplicated = (qseqid != sseqid) & pair_df.duplicated(subset="dup_key", keep="first")
    return blast.loc[~duplicated].reset_index(drop=True)


def build_wt_pairs() -> pd.DataFrame:
    clusters = read_clusters()
    blast = pd.read_csv(RAW / "ALL_SEQS_blastp_named.tsv", sep="\t")
    blast["qseqid"] = blast["qseqid"].str.split("_").str[0]
    blast["sseqid"] = blast["sseqid"].str.split("_").str[0]
    wt_uid_to_cluster = clusters.set_index("wt_uid")["cluster"]
    blast["qseq_cluster"] = blast["qseqid"].map(wt_uid_to_cluster)
    blast["sseq_cluster"] = blast["sseqid"].map(wt_uid_to_cluster)
    blast.drop_duplicates(subset=["qseqid", "sseqid"], inplace=True)

    self_bitscore = blast[blast["qseqid"] == blast["sseqid"]].set_index("qseqid")["bitscore"]
    denominator = 0.5 * (blast["qseqid"].map(self_bitscore) + blast["sseqid"].map(self_bitscore))
    blast["bitscore_norm"] = blast["bitscore"] / denominator
    blast = desymmetrise(blast)

    embeddings = pd.read_csv(RAW / "wt_embeddings.tsv", sep="\t", usecols=["wt_uid", "avg_wt_emb"])
    lookup = {
        uid: np.fromstring(text.strip()[1:-1], sep=",")
        for uid, text in zip(embeddings["wt_uid"], embeddings["avg_wt_emb"])
    }
    blast["avg_emb_dist"] = [
        float(np.sum((lookup[q] - lookup[s]) ** 2)) for q, s in zip(blast["qseqid"], blast["sseqid"])
    ]
    # NaN == NaN is False: a wild type missing from the cluster map is never "same cluster".
    blast["same_cluster"] = blast["qseq_cluster"] == blast["sseq_cluster"]

    pairs = blast[
        ["qseqid", "sseqid", "qseq_cluster", "sseq_cluster", "same_cluster", "bitscore_norm", "avg_emb_dist"]
    ]
    n_same = int(pairs["same_cluster"].sum())
    print(f"wt pairs: {len(pairs)} ({n_same} same cluster, {len(pairs) - n_same} different)")
    return pairs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_mutations().to_csv(OUT / "mutations.tsv", sep="\t", index=False)
    build_wt_pairs().to_csv(OUT / "wt_pairs.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
