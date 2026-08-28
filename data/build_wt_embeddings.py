"""Regenerate ``data/sources/wt_embeddings.tsv`` — mean-pooled ESM-2 embeddings per wild type.

Exp 2.3 bins wild-type pairs by ``d_esm``, the squared L2 distance between the two mean-pooled
embeddings in this file. The file is not a third-party dataset: it is derived here.

    python data/build_wt_embeddings.py                       # writes data/sources/wt_embeddings.tsv
    python data/build_wt_embeddings.py --out /tmp/emb.tsv    # write elsewhere

The wild-type set is the distinct ``qseqid`` values in ``ALL_SEQS_blastp_named.tsv``. Pooling
is a mean over the attention mask, so the BOS and EOS tokens are included in the mean.

The BLAST table names **883** wild types; **841** of them have a sequence in ``data/sources/``
and are embedded here. The other 42 carry no training data, so nothing downstream samples
them. A rebuilt file is therefore 42 rows shorter than the Zenodo copy, which is the one the
paper used; rebuilding is optional.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel

SOURCES = Path(__file__).resolve().parent / "sources"
ESM_CHECKPOINT = "facebook/esm2_t36_3B_UR50D"
BATCH_SIZE = 20

# Every source that carries a (wt_uid, wt_seq) pair. The union is the sequence lookup.
SEQUENCE_FILES = [
    "mega_smdi.tsv",
    "fireprot.tsv",
    "Ssym.tsv",
    "tmdbs.tsv",
    "S2648.tsv",
    "S669.tsv",
]


def sequence_lookup() -> pd.Series:
    frames = []
    for name in SEQUENCE_FILES:
        frame = pd.read_csv(SOURCES / name, sep="\t", usecols=["wt_uid", "wt_seq"], low_memory=False)
        frames.append(frame)
    return pd.concat(frames).drop_duplicates("wt_uid").set_index("wt_uid")["wt_seq"]


def wild_types() -> pd.DataFrame:
    blast = pd.read_csv(SOURCES / "ALL_SEQS_blastp_named.tsv", sep="\t", usecols=["qseqid"])
    uids = blast["qseqid"].str.split("_").str[0].unique()
    table = pd.DataFrame({"wt_uid": uids})
    table["wt_seq"] = table["wt_uid"].map(sequence_lookup())
    missing = int(table["wt_seq"].isna().sum())
    if missing:
        print(f"skipping {missing} of {len(table)} wild types with no sequence in data/sources/")
    return table.dropna(subset=["wt_seq"]).reset_index(drop=True)


def embed(sequences: list[str], device: torch.device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(ESM_CHECKPOINT, use_fast=True)
    model = EsmModel.from_pretrained(ESM_CHECKPOINT, add_pooling_layer=False)
    model.eval()
    model.to(device)

    pooled = []
    for start in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[start : start + BATCH_SIZE]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=5000)
        tokens = tokens.to(device)
        with torch.no_grad():
            hidden = model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        means = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        pooled.append(means.cpu().numpy())
        print(f"  {min(start + BATCH_SIZE, len(sequences))}/{len(sequences)}", flush=True)
    return np.concatenate(pooled, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=SOURCES / "wt_embeddings.tsv")
    args = parser.parse_args()

    table = wild_types()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"embedding {len(table)} wild types with {ESM_CHECKPOINT} on {device}")

    vectors = embed(list(table["wt_seq"]), device)
    # Written as a Python list repr; the readers parse it with np.fromstring on the interior.
    table["avg_wt_emb"] = [list(map(float, row)) for row in vectors]
    table.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out} — {len(table)} rows, {vectors.shape[1]} dimensions")


if __name__ == "__main__":
    main()
