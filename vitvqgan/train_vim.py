import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl

from vitvqgan.utils.general import (
    get_config_from_file,
    initialize_from_config,
    setup_callbacks,
)

here = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    from hiq import print_model, deterministic

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="imagenet_vitvq_small")
    parser.add_argument("-nn", "--num_nodes", type=int, default=1)
    parser.add_argument("-ng", "--num_gpus", type=int, default=1)
    parser.add_argument("-u", "--update_every", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-lr", "--base_lr", type=float, default=4.5e-6)
    parser.add_argument("-a", "--use_amp", default=False, action="store_true")
    parser.add_argument("-b", "--batch_frequency", type=int, default=750)
    parser.add_argument("-m", "--max_images", type=int, default=4)
    args = parser.parse_args()

    # Load configuration
    config = get_config_from_file(f"{here}" / Path("configs") / (args.config + ".yaml"))
    exp_config = OmegaConf.create(
        {
            "name": args.config,
            "epochs": args.epochs,
            "update_every": args.update_every,
            "base_lr": args.base_lr,
            "use_amp": args.use_amp,
            "batch_frequency": args.batch_frequency,
            "max_images": args.max_images,
        }
    )

    # Build model
    model = initialize_from_config(config.model)
    print_model(model)

    model.learning_rate = exp_config.base_lr

    # Setup callbacks
    callbacks, logger = setup_callbacks(exp_config, config)

    # Build data modules
    data = initialize_from_config(config.dataset)
    data.prepare_data()

    # Build trainer
    trainer = pl.Trainer(
        max_epochs=exp_config.epochs,
        precision=16 if exp_config.use_amp else 32,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        num_nodes=args.num_nodes,
        accumulate_grad_batches=exp_config.update_every,
        logger=logger,
    )

    # Train
    trainer.fit(model, data)
