"""Fetch the two source files that are too large to commit.

Everything else this project reads lives in ``data/sources/`` and ships with the repository.
Two files do not:

    mega_smdi.tsv       141 MB   the mega-scale corpus, over GitHub's per-file limit
    wt_embeddings.tsv    99 MB   mean-pooled ESM-2 3B embeddings, regenerable but slow

Both are published in a Zenodo record alongside the paper. This script downloads them into
``data/sources/`` and verifies their SHA-256.

    python data/fetch_raw_data.py            # download whatever is missing, then verify
    python data/fetch_raw_data.py --verify   # verify what is already on disk, download nothing
    python data/fetch_raw_data.py --force    # re-download even if present

``wt_embeddings.tsv`` can also be rebuilt locally with ``data/build_wt_embeddings.py``. The
checksum below is of the published file; a local rebuild will not necessarily reproduce it
byte for byte, since float formatting depends on the torch build and the hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

SOURCES = Path(__file__).resolve().parent / "sources"

# https://doi.org/10.5281/zenodo.22207909
ZENODO_RECORD = 22207909

ZENODO_API = "https://zenodo.org/api/records/{record}"

FILES = {
    "mega_smdi.tsv": {
        "sha256": "a7bb656ec0d02d82783f5c734918395a28fe9be2f87e70f2fb79107129817cb3",
        "size": 140986741,
    },
    "wt_embeddings.tsv": {
        "sha256": "db806ec571fadaaf698e533ad73bac36c247f1e5ae8b7df7024b9a33ae939387",
        "size": 99115952,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zenodo_urls() -> dict[str, str]:
    with urllib.request.urlopen(ZENODO_API.format(record=ZENODO_RECORD)) as response:
        record = json.load(response)
    return {entry["key"]: entry["links"]["self"] for entry in record["files"]}


def download(name: str, url: str, destination: Path) -> None:
    print(f"downloading {name} from {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)


def verify(name: str, spec: dict) -> bool:
    path = SOURCES / name
    if not path.exists():
        print(f"MISSING  {name}")
        return False
    if path.stat().st_size != spec["size"]:
        print(f"FAIL     {name} — size {path.stat().st_size}, expected {spec['size']}")
        return False
    actual = sha256(path)
    if actual != spec["sha256"]:
        print(f"FAIL     {name} — sha256 {actual}, expected {spec['sha256']}")
        return False
    print(f"OK       {name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="verify only, download nothing")
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    args = parser.parse_args()

    if not args.verify:
        missing = [n for n in FILES if args.force or not (SOURCES / n).exists()]
        if missing:
            urls = zenodo_urls()
            for name in missing:
                if name not in urls:
                    raise SystemExit(f"{name} is not in Zenodo record {ZENODO_RECORD}")
                download(name, urls[name], SOURCES / name)

    ok = all([verify(name, spec) for name, spec in FILES.items()])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
