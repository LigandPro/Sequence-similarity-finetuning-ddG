"""Run every config of one experiment, in order, on one GPU.

    python -m thermality.run_experiment experiments/exp2_bitscore_norm

Pretraining configs run first.  The fine-tuning configs start from the last checkpoint under
their ``FLEET.PRETRAINED_CHECKPOINT``, which is the directory the pretraining run writes.

Set ``CUDA_VISIBLE_DEVICES`` to choose the GPU.  There is no fan-out: the configs run one
after another.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from thermality.config import run_identity


def load_configs(experiment_dir):
    """Every config in the experiment, pretraining first, keyed by run name."""
    configs = sorted((experiment_dir / "configs").rglob("*.yaml"))
    if not configs:
        raise ValueError(f"no configs under {experiment_dir}/configs")

    by_run_name = {}
    for path in configs:
        _, run_name, _ = run_identity(path)
        if run_name in by_run_name:
            raise ValueError(f"{path} and {by_run_name[run_name]} both resolve to run {run_name!r}")
        by_run_name[run_name] = path

    is_finetune = {p: "FLEET" in (yaml.safe_load(p.read_text()) or {}) for p in configs}
    return sorted(by_run_name.items(), key=lambda item: (is_finetune[item[1]], item[0])), is_finetune


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", type=Path, help="an experiments/<name>/ directory")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    experiment_dir = args.experiment.resolve()
    ordered, is_finetune = load_configs(experiment_dir)

    print(f"{experiment_dir.name}: {len(ordered)} configs")
    for index, (run_name, path) in enumerate(ordered, 1):
        entry_point = "thermality.finetune" if is_finetune[path] else "thermality.train"
        command = [sys.executable, "-m", entry_point, "-c", str(path)]

        print(f"\n[{index}/{len(ordered)}] {run_name}\n  {' '.join(command)}")
        if args.dry_run:
            continue
        result = subprocess.run(command)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
