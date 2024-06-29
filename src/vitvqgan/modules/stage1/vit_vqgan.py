from typing import List, Tuple, Dict, Any, Optional
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl

from vitvqgan.modules.stage1.layers import ViTEncoder as Encoder, ViTDecoder as Decoder
from vitvqgan.modules.stage1.quantizers import VectorQuantizer, GumbelQuantizer
from vitvqgan.utils.general import initialize_from_config


class ViTVQ(pl.LightningModule):
    def __init__(
        self,
        image_key: str,
        image_size: int,
        patch_size: int,
        encoder: OmegaConf,
        decoder: OmegaConf,
        quantizer: OmegaConf,
        loss: OmegaConf,
        path: Optional[str] = None,
        ignore_keys: List[str] = list(),
        scheduler: Optional[OmegaConf] = None,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.path = path
        self.ignore_keys = ignore_keys
        self.image_key = image_key
        self.scheduler = scheduler

        self.loss = initialize_from_config(loss)
        self.encoder = Encoder(image_size=image_size, patch_size=patch_size, **encoder)
        self.decoder = Decoder(image_size=image_size, patch_size=patch_size, **decoder)
        self.quantizer = VectorQuantizer(**quantizer)
        self.pre_quant = nn.Linear(encoder.dim, quantizer.embed_dim)
        self.post_quant = nn.Linear(quantizer.embed_dim, decoder.dim)

        if path is not None:
            self.init_from_ckpt(path, ignore_keys)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        quant, diff = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()):
        sd = torch.load(path, map_location="cuda:0")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(
        self, x: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h = self.encoder(x)
        h = self.pre_quant(h)
        quant, emb_loss, _ = self.quantizer(h)

        return quant, emb_loss

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant(quant)
        dec = self.decoder(quant)

        return dec

    def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
        h = self.encoder(x)
        h = self.pre_quant(h)
        _, _, codes = self.quantizer(h)

        return codes

    def decode_codes(self, code: torch.LongTensor) -> torch.FloatTensor:
        quant = self.quantizer.embedding(code)
        quant = self.quantizer.norm(quant)

        if self.quantizer.use_residual:
            quant = quant.sum(-2)

        dec = self.decode(quant)

        return dec

    def get_input(self, batch: Tuple[Any, Any], key: str = "image") -> Any:
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.dtype == torch.double:
            x = x.float()

        return x.contiguous()

    def training_step(
        self, batch: Tuple[Any, Any], batch_idx: int
    ) -> torch.FloatTensor:
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        opt_g, opt_d = self.optimizers()

        # Perform generator update
        if self.global_step % 2 == 0:
            ll = self.decoder.get_last_layer()
            aeloss, log_dict_ae = self.loss.forward_generator(
                qloss,
                x,
                xrec,
                self.global_step,
                batch_idx,
                last_layer=ll,
                split="train",
            )

            # Ensure loss is not None
            if aeloss is None or log_dict_ae is None:
                raise ValueError(
                    "The loss function returned None. Please check the implementation of the loss function."
                )

            self.log(
                "train/total_loss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            del log_dict_ae["train/total_loss"]
            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )

            self.manual_backward(aeloss)
            opt_g.step()
            opt_g.zero_grad()

        # Perform discriminator update
        else:
            ll = self.decoder.get_last_layer()
            discloss, log_dict_disc = self.loss.forward_discriminator(
                qloss,
                x,
                xrec,
                self.global_step,
                batch_idx,
                last_layer=ll,
                split="train",
            )

            if discloss is None or log_dict_disc is None:
                raise ValueError(
                    "The loss function returned None. Please check the implementation of the loss function."
                )

            self.log(
                "train/disc_loss",
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            del log_dict_disc["train/disc_loss"]
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )

            self.manual_backward(discloss)
            opt_d.step()
            opt_d.zero_grad()

        # Return the appropriate loss
        # return aeloss if self.global_step % 2 == 0 else discloss

    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> Dict:
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss.forward_generator(
            qloss,
            x,
            xrec,
            self.global_step,
            batch_idx,
            last_layer=self.decoder.get_last_layer(),
            split="val",
        )

        rec_loss = log_dict_ae["val/rec_loss"]

        self.log(
            "val/rec_loss",
            rec_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/total_loss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        del log_dict_ae["val/rec_loss"]
        del log_dict_ae["val/total_loss"]

        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True
        )

        if hasattr(self.loss, "discriminator"):
            discloss, log_dict_disc = self.loss.forward_discriminator(
                qloss,
                x,
                xrec,
                self.global_step,
                batch_idx,
                last_layer=self.decoder.get_last_layer(),
                split="val",
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True
            )
        return log_dict_ae

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate
        optim_groups = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.pre_quant.parameters())
            + list(self.post_quant.parameters())
            + list(self.quantizer.parameters())
        )

        optimizers = [
            torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
        ]
        schedulers = []

        if hasattr(self.loss, "discriminator"):
            optimizers.append(
                torch.optim.AdamW(
                    self.loss.discriminator.parameters(),
                    lr=lr,
                    betas=(0.9, 0.99),
                    weight_decay=1e-4,
                )
            )

        if self.scheduler is not None:
            self.scheduler.params.start = lr
            scheduler = initialize_from_config(self.scheduler)

            schedulers = [
                {
                    "scheduler": lr_scheduler.LambdaLR(
                        optimizer, lr_lambda=scheduler.schedule
                    ),
                    "interval": "step",
                    "frequency": 1,
                }
                for optimizer in optimizers
            ]

        return optimizers, schedulers

    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        quant, _ = self.encode(x)

        log["originals"] = x
        log["reconstructions"] = self.decode(quant)

        return log


class ViTVQGumbel(ViTVQ):
    def __init__(
        self,
        image_key: str,
        image_size: int,
        patch_size: int,
        encoder: OmegaConf,
        decoder: OmegaConf,
        quantizer: OmegaConf,
        loss: OmegaConf,
        path: Optional[str] = None,
        ignore_keys: List[str] = list(),
        temperature_scheduler: OmegaConf = None,
        scheduler: Optional[OmegaConf] = None,
    ) -> None:
        super().__init__(
            image_key,
            image_size,
            patch_size,
            encoder,
            decoder,
            quantizer,
            loss,
            None,
            None,
            scheduler,
        )

        self.temperature_scheduler = (
            initialize_from_config(temperature_scheduler)
            if temperature_scheduler
            else None
        )
        self.quantizer = GumbelQuantizer(**quantizer)

        if path is not None:
            self.init_from_ckpt(path, ignore_keys)

    def training_step(
        self, batch: Tuple[Any, Any], batch_idx: int
    ) -> torch.FloatTensor:
        if self.temperature_scheduler:
            self.quantizer.temperature = self.temperature_scheduler(self.global_step)
        loss = super().training_step(batch, batch_idx)
        self.log(
            "temperature",
            self.quantizer.temperature,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return loss


small_config = OmegaConf.create(
    {
        "image_key": "image",
        "image_size": 256,
        "patch_size": 8,
        "encoder": {"dim": 512, "depth": 8, "heads": 8, "mlp_dim": 2048},
        "decoder": {"dim": 512, "depth": 8, "heads": 8, "mlp_dim": 2048},
        "quantizer": {"embed_dim": 32, "n_embed": 8192},
        "loss": {
            "target": "vitvqgan.losses.vqperceptual.VQLPIPSWithDiscriminator",
            "params": {
                "loglaplace_weight": 0.0,
                "loggaussian_weight": 1.0,
                "perceptual_weight": 0.1,
                "adversarial_weight": 0.1,
            },
        },
    }
)

base_config = OmegaConf.merge(
    small_config,
    {
        "encoder": {"dim": 768, "depth": 12, "heads": 12, "mlp_dim": 3072},
        "decoder": {"dim": 768, "depth": 12, "heads": 12, "mlp_dim": 3072},
    },
)

large_config = OmegaConf.merge(
    small_config,
    {
        "encoder": {"dim": 512, "depth": 8, "heads": 8, "mlp_dim": 2048},
        "decoder": {"dim": 1280, "depth": 32, "heads": 16, "mlp_dim": 5120},
    },
)

if __name__ == "__main__":
    import torch
    from omegaconf import OmegaConf
    from skimage import data
    from skimage.transform import resize
    import matplotlib.pyplot as plt

    # Define the configuration using OmegaConf
    config = base_config
    # Initialize the model with the given config
    model = ViTVQ(
        image_key=config.image_key,
        image_size=config.image_size,
        patch_size=config.patch_size,
        encoder=config.encoder,
        decoder=config.decoder,
        quantizer=config.quantizer,
        loss=config.loss,
    )

    # Check if a GPU is available and move the model to the GPU if it is
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load model checkpoint
    checkpoint_path = "mbin/imagenet_vitvq_base.ckpt"
    model.init_from_ckpt(checkpoint_path)

    # Set the model to evaluation mode
    model.eval()

    # Load and preprocess the coffee image
    image = data.coffee()
    image_resized = resize(image, (256, 256), anti_aliasing=True)
    input_data = (
        torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    )

    # Run inference
    with torch.no_grad():
        output, extra = model(input_data)
        output_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Plot the input and output images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_resized)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(output_image)
    axes[1].set_title("Output Image")
    axes[1].axis("off")
    plt.savefig("base.png")
    plt.show()
