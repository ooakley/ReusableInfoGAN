"""Contains class for handling save/load logic, training and inference for the infoGAN model."""
import os
import json
from datetime import datetime
from typing import Tuple, List

import torch
import torch.nn as nn
import numpy as np

from . import model


class InfoGANHandler:
    """Encapsulates infoGAN training."""

    def __init__(self, config_dict: dict, use_logs: bool = True) -> None:
        # Initial seed number:
        torch.set_deterministic(True)
        self.seed = config_dict["seed"]

        # Initialising model parameters:
        self._init_networks(config_dict)
        self.zdim = config_dict["zdim"]

        # Initialising discriminator training parameters:
        self.discriminator_params = \
            list(self.discriminator.conv.parameters()) + list(self.class_head.linear.parameters())
        disc_lr = config_dict["learning_rate"] * config_dict["discopt_lr_ratio"]
        self.discriminator_opt = torch.optim.Adam(self.discriminator_params, lr=disc_lr)
        self.bce_logits_loss = nn.BCEWithLogitsLoss(reduction="none")

        # Initialising generator training parameters:
        self.generator_params = \
            list(self.generator.linear.parameters()) + list(self.generator.conv.parameters()) + \
            list(self.aux_head.linear.parameters())
        self.generator_opt = torch.optim.Adam(
            self.generator_params, lr=config_dict["learning_rate"]
        )
        self.mse_loss = nn.MSELoss(reduction="none")

        # Configuring logging:
        if use_logs:
            self._set_logs(config_dict)
        self.loss_history: List[List[float]] = []
        self.epochs_performed = 0

    def _set_seed(self, seed: int = None) -> None:
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _set_logs(self, config_dict: dict) -> None:
        log_dir = "logs"
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        self.date_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        log_path = os.path.join(log_dir, f"{self.date_str}-{config_dict['config_name']}")
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        self.log_path = log_path
        with open(os.path.join(log_path, "cfg.json"), "w") as f:
            json.dump(config_dict, f)

    def _init_networks(self, config_dict: dict) -> None:
        # Initialise generator:
        self._set_seed()
        self.generator = model.GeneratorNetwork(config_dict["zdim"], config_dict["generator"])
        self.generator.linear.apply(model.relu_kaiming_init_weights)
        self.generator.conv.apply(model.relu_kaiming_init_weights)

        # Initialise discriminator:
        self._set_seed()
        self.discriminator = model.DiscriminatorNetwork(config_dict["discriminator"])
        self.discriminator.conv.apply(model.leaky_relu_kaiming_init_weights)

        # Initialise classification head:
        self._set_seed()
        self.class_head = model.ClassificationHead(config_dict["discriminator"])
        self.class_head.linear.apply(model.final_linear_init_weights)

        # Initialise auxiliary head:
        self._set_seed()
        self.aux_head = model.AuxiliaryHead(config_dict)
        self.aux_head.linear.apply(model.leaky_relu_kaiming_init_weights)

    def calculate_loss(
            self, real_images: torch.Tensor, noise_vector: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate full discriminator loss from real image input."""
        # Calculating discriminator loss on real images:
        real_features = self.discriminator(real_images)
        real_probability_logit = self.class_head(real_features)
        real_loss = self.bce_logits_loss(
            real_probability_logit, torch.ones(100, 1)
        ).mean()

        # Generating fake images:
        fake_images = self.generator(noise_vector)

        # Generating discrimimator loss on fake images:
        fake_features = self.discriminator(fake_images)
        fake_probability_logit = self.class_head(fake_features)
        fake_loss = self.bce_logits_loss(
            fake_probability_logit, torch.zeros_like(fake_probability_logit)
        ).mean()

        disc_class_loss = 0.5 * (real_loss + fake_loss)

        # Calculating discriminator regularisation term:
        disc_regularisation = self._calculate_discriminator_regularisation(
            real_images, real_probability_logit,
            fake_images, fake_probability_logit
        )

        # Generator classification loss - note the change to torch.ones_like:
        gen_class_loss = self.bce_logits_loss(
            fake_probability_logit, torch.ones_like(fake_probability_logit)
        ).mean()

        # Generator info loss:
        predicted_codes = self.aux_head(fake_features)
        gen_info_loss = self.mse_loss(predicted_codes, noise_vector).mean()

        return disc_class_loss, disc_regularisation, gen_class_loss, gen_info_loss

    def _calculate_discriminator_regularisation(
            self,
            real_images: torch.Tensor, real_logits: torch.Tensor,
            fake_images: torch.Tensor, fake_logits: torch.Tensor
            ) -> torch.Tensor:
        # Calculating gradients of logits w/r/t image inputs:
        real_gradients = torch.autograd.grad(
            real_logits, real_images, grad_outputs=torch.ones_like(real_logits), retain_graph=True
        )[0]
        fake_gradients = torch.autograd.grad(
            fake_logits, fake_images, grad_outputs=torch.ones_like(real_logits), retain_graph=True
        )[0]

        # Calculating the L2 norm of these gradients:
        norm_realgrad = torch.linalg.norm(
            real_gradients.view(real_logits.shape[0], -1), dim=1, keepdim=True
        )
        norm_fakegrad = torch.linalg.norm(
            fake_gradients.view(fake_logits.shape[0], -1), dim=1, keepdim=True
        )

        # Calcaulting actual probabilities:
        real_probability = torch.sigmoid(real_logits)
        fake_probability = torch.sigmoid(fake_logits)

        # Putting it all together:
        real_prob_term = torch.square(1 - real_probability)
        fake_prob_term = torch.square(fake_probability)
        real_sqr_norm = torch.square(norm_realgrad)
        fake_sqr_norm = torch.square(norm_fakegrad)

        real_reg = torch.mul(real_prob_term, real_sqr_norm)
        fake_reg = torch.mul(fake_prob_term, fake_sqr_norm)

        return torch.mean(real_reg + fake_reg, dim=0)

    def train_on_batch(self, real_images: torch.Tensor, noise_vector: torch.Tensor) -> None:
        """Train all networks on batch of images."""
        # Calculating loss:
        disc_class_loss, disc_regularisation, gen_class_loss, gen_info_loss = \
            self.calculate_loss(real_images, noise_vector)

        # Accumulating discriminator gradients:
        discriminator_loss = disc_class_loss + disc_regularisation
        self.discriminator_opt.zero_grad(set_to_none=True)
        discriminator_loss.backward(retain_graph=True)

        # Accumulating generator gradients:
        generator_loss = gen_class_loss + gen_info_loss
        self.generator_opt.zero_grad(set_to_none=True)
        generator_loss.backward()

        # Updating weights
        self.discriminator_opt.step()
        self.generator_opt.step()

        # Updating history:
        self.loss_history.append([
            disc_class_loss.item(), disc_regularisation.item(),
            gen_class_loss.item(), gen_info_loss.item()
        ])

    def train_on_epoch(self, dataset: torch.utils.data.Dataset, batch_size: int = 100) -> None:
        """Train all networks on given dataset for one epoch."""
        self._set_seed(self.seed + self.epochs_performed)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        for real_images in dataloader:
            real_images.requires_grad = True
            noise_vector = torch.randn(batch_size, self.zdim)
            self.train_on_batch(real_images, noise_vector)
        self.epochs_performed += 1
