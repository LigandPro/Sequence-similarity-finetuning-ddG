"""yacs defaults for every experiment.

Config files carry paths relative to the project root — ``data/splits/...``,
``checkpoints/...`` — and :func:`resolve_path` turns them into absolute ones.  Set
``THERMALITY_ROOT`` to run against a tree other than the checkout this package lives in.
"""

import os
from pathlib import Path

from yacs.config import CfgNode as CN

PROJECT_ROOT = Path(os.environ.get("THERMALITY_ROOT", Path(__file__).resolve().parents[2]))


def resolve_path(path):
    """A project-relative path from a config file, as an absolute path."""
    if not path:
        return None
    return str(PROJECT_ROOT / path)


def run_identity(config_path):
    """``(experiment, run_name, relative stem)`` for a config file — all three from its path.

    The path *is* the identity of a run.  Checkpoints mirror the config tree, so
    ``experiments/exp2_d_esm/configs/within/pretrain.yaml`` writes to
    ``checkpoints/exp2_d_esm/within/pretrain/`` and reports as ``within__pretrain``.
    """
    path = Path(config_path).resolve()
    for parent in path.parents:
        if parent.name == "configs":
            stem = path.relative_to(parent).with_suffix("")
            return parent.parent.name, str(stem).replace("/", "__"), stem
    raise ValueError(f"{config_path} is not inside an experiment's configs/ directory")


cfg = CN()

# Model configs
cfg.MODEL = CN()
cfg.MODEL.ESM_CHECKPOINT = "facebook/esm2_t6_8M_UR50D"
cfg.MODEL.ARCHITECTURE = "crossatt_true_siam"

cfg.MODEL.SELFATT_DIM_EXPANSION = 2

cfg.MODEL.PRECODER_LAYERS = 1
cfg.MODEL.DECODER_LAYERS = 2
cfg.MODEL.ENCODER_LAYERS = 1

cfg.MODEL.NUM_HEADS = 2

cfg.MODEL.MLP_LAYERS = 3


# Training configs
cfg.TRAIN = CN()

cfg.TRAIN.DATASET_PATH = ""
cfg.TRAIN.TRUE_RANDOM_SYNTH_AUG = 0.0
cfg.TRAIN.BATCH_SIZE = 200
cfg.TRAIN.MUTATION_TYPES = [["single", 1], ["double", 0], ["synthetic_double", 0], ["triple", 0], ["quadruple", 0], ["zero", 0]] # ['null', 'single', 'double', 'synthetic_double', 'triple', 'quadruple']
cfg.TRAIN.TRUNCATE_LENGTH = 5000

cfg.TRAIN.LR = 1.0e-4
cfg.TRAIN.DROPOUT = 0.0
cfg.TRAIN.GRAD_NORM = 1.0
cfg.TRAIN.GRAD_ACCUM = 1

cfg.TRAIN.SYM_LOSS = 0.0
cfg.TRAIN.ALPHA = 0.0
cfg.TRAIN.MAX_STEPS = 10000  # Total training steps
cfg.TRAIN.EVAL_STEPS = 500  # Evaluate and save every N steps
cfg.TRAIN.EARLY_STOPPING_PATIENCE_STEPS = 2500  # Stop after 2500 steps without improvement

cfg.TRAIN.SCHEDULER = "linear"
cfg.TRAIN.SCHEDULER_KWARGS = ({"patience": 10, "factor": 0.8, "min_lr": 0.5*1e-5} if cfg.TRAIN.SCHEDULER ==
                            "reduce_lr_on_plateau" else \
                                None)

cfg.TRAIN.SEED = 1984

cfg.TEST = CN()
# Format: [name, path, needs_reverse_double]
# multicluster automatically detected: directory = multicluster, file = single dataset
cfg.TEST.DATASETS = [
    ("dataset_name", "dataset_path", False),
]
cfg.TEST.TARGET_DATASET = "val"
cfg.TEST.BATCH_SIZE = 200

# Fine-tuning configuration (one model per cluster)
cfg.FLEET = CN()
cfg.FLEET.PRETRAINED_CHECKPOINT = ""
cfg.FLEET.TRAIN_DIR = ""
cfg.FLEET.VAL_DIR = ""
cfg.FLEET.TEST_DIR = ""

def get_cfg_defaults():
    return cfg.clone()
