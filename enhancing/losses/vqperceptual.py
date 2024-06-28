from omegaconf import OmegaConf
from typing import Optional, Tuple

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *


class DummyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class VQLPIPS(nn.Module):
    def __init__(self, codebook_weight: float = 1.0,
                 loglaplace_weight: float = 1.0,
                 loggaussian_weight: float = 1.0,
                 perceptual_weight: float = 1.0) -> None:
        super().__init__()
        self.perceptual_loss = lpips.LPIPS(net="vgg", verbose=False)

        self.codebook_weight = codebook_weight
        self.loglaplace_weight = loglaplace_weight
        self.loggaussian_weight = loggaussian_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, codebook_loss: torch.FloatTensor, inputs: torch.FloatTensor, reconstructions: torch.FloatTensor,
                global_step: int, batch_idx: int, last_layer: Optional[nn.Module] = None,
                split: Optional[str] = "train") -> Tuple:
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        loglaplace_loss = (reconstructions - inputs).abs().mean()
        loggaussian_loss = (reconstructions - inputs).pow(2).mean()
        perceptual_loss = self.perceptual_loss(inputs * 2 - 1, reconstructions * 2 - 1).mean()

        nll_loss = self.loglaplace_weight * loglaplace_loss + self.loggaussian_weight * loggaussian_loss + self.perceptual_weight * perceptual_loss
        loss = nll_loss + self.codebook_weight * codebook_loss

        log = {"{}/total_loss".format(split): loss.clone().detach(),
               "{}/quant_loss".format(split): codebook_loss.detach(),
               "{}/rec_loss".format(split): nll_loss.detach(),
               "{}/loglaplace_loss".format(split): loglaplace_loss.detach(),
               "{}/loggaussian_loss".format(split): loggaussian_loss.detach(),
               "{}/perceptual_loss".format(split): perceptual_loss.detach()
               }

        return loss, log


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start: int = 0,
                 disc_loss: str = 'vanilla',
                 disc_params: Optional[OmegaConf] = dict(),
                 codebook_weight: float = 1.0,
                 loglaplace_weight: float = 1.0,
                 loggaussian_weight: float = 1.0,
                 perceptual_weight: float = 1.0,
                 adversarial_weight: float = 1.0,
                 use_adaptive_adv: bool = False,
                 r1_gamma: float = 10,
                 do_r1_every: int = 16) -> None:

        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "least_square"], f"Unknown GAN loss '{disc_loss}'."
        self.perceptual_loss = lpips.LPIPS(net="vgg", verbose=False)

        self.codebook_weight = codebook_weight
        self.loglaplace_weight = loglaplace_weight
        self.loggaussian_weight = loggaussian_weight
        self.perceptual_weight = perceptual_weight

        self.discriminator = StyleDiscriminator(**disc_params)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "least_square":
            self.disc_loss = least_square_d_loss

        self.adversarial_weight = adversarial_weight
        self.use_adaptive_adv = use_adaptive_adv
        self.r1_gamma = r1_gamma
        self.do_r1_every = do_r1_every

    def calculate_adaptive_factor(self, nll_loss: torch.FloatTensor,
                                  g_loss: torch.FloatTensor, last_layer: nn.Module) -> torch.FloatTensor:
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        adapt_factor = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        adapt_factor = adapt_factor.clamp(0.0, 1e4).detach()

        return adapt_factor

    def forward_generator(self, codebook_loss: torch.FloatTensor, inputs: torch.FloatTensor, reconstructions: torch.FloatTensor,
                global_step: int, batch_idx: int = 0, last_layer: Optional[nn.Module] = None,
                split: Optional[str] = "train") -> Tuple:
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        # Generator update
        loglaplace_loss = (reconstructions - inputs).abs().mean()
        loggaussian_loss = (reconstructions - inputs).pow(2).mean()
        perceptual_loss = self.perceptual_loss(inputs * 2 - 1, reconstructions * 2 - 1).mean()

        nll_loss = self.loglaplace_weight * loglaplace_loss + self.loggaussian_weight * loggaussian_loss + self.perceptual_weight * perceptual_loss

        logits_fake = self.discriminator(reconstructions)
        g_loss = self.disc_loss(logits_fake)

        try:
            d_weight = self.adversarial_weight
            if self.use_adaptive_adv:
                d_weight *= self.calculate_adaptive_factor(nll_loss, g_loss, last_layer=last_layer)
        except RuntimeError:
            assert not self.training
            d_weight = torch.tensor(0.0)

        disc_factor = 1 if global_step >= self.discriminator_iter_start else 0
        loss = nll_loss + disc_factor * d_weight * g_loss + self.codebook_weight * codebook_loss

        log = {
            f"{split}/total_loss": loss.clone().detach(),
            f"{split}/quant_loss": codebook_loss.detach(),
            f"{split}/rec_loss": nll_loss.detach(),
            f"{split}/loglaplace_loss": loglaplace_loss.detach(),
            f"{split}/loggaussian_loss": loggaussian_loss.detach(),
            f"{split}/perceptual_loss": perceptual_loss.detach(),
            f"{split}/g_loss": g_loss.detach()
        }

        if self.use_adaptive_adv:
            log[f"{split}/d_weight"] = d_weight.detach()

        return loss, log

    def forward_discriminator(self, codebook_loss: torch.FloatTensor, inputs: torch.FloatTensor, reconstructions: torch.FloatTensor,
                global_step: int, batch_idx: int = 0, last_layer: Optional[nn.Module] = None,
                split: Optional[str] = "train") -> Tuple:
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        # Discriminator update
        disc_factor = 1 if global_step >= self.discriminator_iter_start else 0
        do_r1 = self.training and bool(disc_factor) and batch_idx % self.do_r1_every == 0

        logits_real = self.discriminator(inputs.requires_grad_(do_r1))
        logits_fake = self.discriminator(reconstructions.detach())

        d_loss = disc_factor * self.disc_loss(logits_fake, logits_real)
        if do_r1:
            gradients, = torch.autograd.grad(outputs=logits_real.sum(), inputs=inputs, create_graph=True)
            gradients_norm = gradients.square().sum([1, 2, 3]).mean()
            d_loss += self.r1_gamma * self.do_r1_every * gradients_norm / 2

        log = {
            f"{split}/disc_loss": d_loss.detach(),
            f"{split}/logits_real": logits_real.detach().mean(),
            f"{split}/logits_fake": logits_fake.detach().mean()
        }

        if do_r1:
            log[f"{split}/r1_reg"] = gradients_norm.detach()

        return d_loss, log

