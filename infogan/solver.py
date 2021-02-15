"""Contains class for handling save/load logic, training and inference for the infoGAN model."""
import os
import copy
import json
from datetime import datetime
from typing import Tuple, List, Union, Iterator

import torch
import torch.nn as nn
import numpy as np

from infogan import model, initialisations


class InfoGANHandler:
    """Encapsulates infoGAN training."""

    def __init__(
            self, config_dict: dict, use_logs: bool = True,
            device: Union[str, torch.device] = "cpu") -> None:
        # Initial seed number:
        torch.set_deterministic(True)
        self.seed = config_dict["seed"]
        self.device = device

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
        self.discriminator_grad_history: List[List[Tuple[str, bool, torch.Tensor]]] = []
        self.generator_grad_history: List[List[Tuple[str, bool, torch.Tensor]]] = []

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
            json.dump(config_dict, f, indent=4)

    def _init_networks(self, config_dict: dict) -> None:
        # Initialise generator:
        self._set_seed()
        self.generator = model.GeneratorNetwork(config_dict["zdim"], config_dict["generator"])
        self.generator.linear.apply(initialisations.gen_linear_init_weights)
        self.generator.conv.apply(initialisations.gen_relu_init_weights)
        initialisations.gen_tanh_init_weights(self.generator.conv[-2])
        self.generator.to(self.device)

        # Initialise discriminator:
        self._set_seed()
        self.discriminator = model.DiscriminatorNetwork(config_dict["discriminator"])
        self.discriminator.conv.apply(initialisations.disc_lrelu_init_weights)
        self.discriminator.to(self.device)

        # Initialise classification head:
        self._set_seed()
        self.class_head = model.ClassificationHead(config_dict["discriminator"])
        self.class_head.linear.apply(initialisations.disc_sigmoid_init_weights)
        self.class_head.to(self.device)

        # Initialise auxiliary head:
        self._set_seed()
        self.aux_head = model.AuxiliaryHead(config_dict)
        self.aux_head.linear.apply(lambda x: initialisations.disc_lrelu_init_weights(x, alpha=0.2))
        initialisations.disc_sigmoid_init_weights(self.aux_head.linear[-1], gain=0.3)
        self.aux_head.to(self.device)

    def calculate_discriminator_loss(
            self, real_images: torch.Tensor, fake_images: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate full discriminator loss from real image input."""
        # Calculating discriminator loss on real images:
        real_features = self.discriminator(real_images)
        real_probability_logit = self.class_head(real_features)
        real_loss = self.bce_logits_loss(
            real_probability_logit, torch.ones_like(real_probability_logit)
        ).mean()

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

        return disc_class_loss, disc_regularisation

    def _calculate_discriminator_regularisation(
            self,
            real_images: torch.Tensor, real_logits: torch.Tensor,
            fake_images: torch.Tensor, fake_logits: torch.Tensor
            ) -> torch.Tensor:
        # Calculating gradients of logits w/r/t image inputs:
        real_gradients = torch.autograd.grad(
            real_logits, real_images, grad_outputs=torch.ones_like(real_logits),
            retain_graph=True,
            create_graph=True
        )[0]
        fake_gradients = torch.autograd.grad(
            fake_logits, fake_images, grad_outputs=torch.zeros_like(real_logits),
            retain_graph=True,
            create_graph=True
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

    def calculate_generator_loss(
            self, fake_images: torch.Tensor, noise_vector: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate full generator loss from fake image input."""
        # Generator classification loss - note the change to torch.ones_like:
        fake_features = self.discriminator(fake_images)
        fake_probability_logit = self.class_head(fake_features)
        gen_class_loss = self.bce_logits_loss(
            fake_probability_logit, torch.ones_like(fake_probability_logit)
        ).mean()

        # Generator info loss:
        predicted_codes = self.aux_head(fake_features)
        gen_info_loss = self.mse_loss(predicted_codes, noise_vector).mean()

        return gen_class_loss, gen_info_loss

    def train_on_batch(self, real_images: torch.Tensor, noise_vector: torch.Tensor) -> None:
        """Train all networks on batch of images."""
        # Generating fake images:
        fake_images = self.generator(noise_vector)

        # Calculating discriminator loss:
        disc_class_loss, disc_regularisation = \
                self.calculate_discriminator_loss(real_images, fake_images)

        # Accumulating discriminator gradients:
        discriminator_loss = disc_class_loss + disc_regularisation
        self.discriminator_opt.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        self.discriminator_opt.step()

        # Detaching & recording gradients:
        if self.epochs_performed == 0:
            self.discriminator_grad_history.append(
                [
                    copy.deepcopy((n, p.requires_grad, p.grad.detach().abs().mean().cpu())) \
                    for n, p in self.discriminator.named_parameters()
                ]
            )

        # Accumulating generator gradients:
        noise_vector.requires_grad = True
        fake_images = self.generator(noise_vector)
        gen_class_loss, gen_info_loss = \
            self.calculate_generator_loss(fake_images, noise_vector)
        generator_loss = gen_class_loss + gen_info_loss
        self.generator_opt.zero_grad(set_to_none=True)
        generator_loss.backward()
        self.generator_opt.step()

        # Detaching and recording gradients:
        if self.epochs_performed == 0:
            self.generator_grad_history.append(
                [
                    copy.deepcopy((n, p.requires_grad, p.grad.detach().abs().mean().cpu())) \
                    for n, p in self.generator.named_parameters()
                ]
            )

        # Updating history:
        self.loss_history.append([
            disc_class_loss.item(), disc_regularisation.item(),
            gen_class_loss.item(), gen_info_loss.item()
        ])

    def train_on_epoch(self, dataset: torch.utils.data.Dataset, batch_size: int = 100) -> None:
        """Train all networks on given dataset for one epoch."""
        self._set_seed(self.seed + self.epochs_performed)

        # Initialising dataloader:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        # Iterating through batches of real images:
        for real_images in dataloader:
            real_images = real_images.to(self.device)
            real_images.requires_grad = True
            noise_vector = torch.randn(real_images.shape[0], self.zdim)
            noise_vector = noise_vector.to(self.device)
            self.train_on_batch(real_images, noise_vector)
        
        # Updating epoch tracker:
        self.epochs_performed += 1
