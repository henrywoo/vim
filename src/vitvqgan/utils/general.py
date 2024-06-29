# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------
import random
import importlib
import pathlib
from typing import List, ClassVar
from datetime import datetime

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from .callback import *


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_obj_from_str(name: str, reload: bool = False) -> ClassVar:
    module, cls = name.rsplit(".", 1)

    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)


def initialize_from_config(config: OmegaConf) -> object:
    t = config["target"]
    d = config.get("params", dict())
    o = get_obj_from_str(t)
    return o(**d)


def setup_callbacks(
    exp_config: OmegaConf, config: OmegaConf
) -> Tuple[List[Callback], WandbLogger]:
    now = datetime.now().strftime("%d%m%Y_%H%M%S")
    basedir = pathlib.Path("experiments", exp_config.name, now)
    os.makedirs(basedir, exist_ok=True)

    setup_callback = SetupCallback(config, exp_config, basedir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename=exp_config.name + "-{epoch:02d}",
        monitor="train/total_loss",
        save_top_k=-1,
        verbose=False,
    )
    os.makedirs(setup_callback.logdir / "wandb", exist_ok=True)
    logger = WandbLogger(
        save_dir=str(setup_callback.logdir), name=exp_config.name + "_" + str(now)
    )
    logger_img_callback = ImageLogger(exp_config.batch_frequency, exp_config.max_images)

    return [setup_callback, checkpoint_callback, logger_img_callback], logger


def get_config_from_file(config_file: str) -> Dict:
    config_file = OmegaConf.load(config_file)
    return config_file
