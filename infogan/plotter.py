"""Class for methods useful for ouputting metrics related to infoGAN performance."""
import os
from datetime import datetime
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from .solver import InfoGANHandler


class InfoGANPlotter:
    """Class to encapsulate methods for generate infoGAN metrics."""

    def __init__(self, model: InfoGANHandler) -> None:
        """Initialise plotter with trained model."""
        self.model = model
        self.log_path = model.log_path
        self.loss_history = model.loss_history
        self.metadata = {
            "user": os.getlogin(),
            "date": datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            "model": f"infogan-{model.date_str}",
            "source_code": "https://github.com/ooakley/ReusableInfoGAN"
        }

    def _extract_loss(self, index: int) -> List[np.ndarray]:
        loss = [np.array([batch[index]]) for batch in self.loss_history]
        loss = np.concatenate(loss, axis=0)
        return loss

    def plot_loss_history(self) -> None:
        """Plot training dynamics with different losses."""
        # Generating numpy arrays:
        disc_class_loss = self._extract_loss(0)
        disc_regularisation = self._extract_loss(1)
        gen_class_loss = self._extract_loss(2)
        info_loss = self._extract_loss(3)

        # Generating info/discreg plot:
        fig, ax = plt.subplots()
        ax.plot(disc_regularisation)
        ax.plot(info_loss)
        plt.legend(
            ("Disc Regularisation", "Info Loss"),
            loc='upper right')
        plt.title("Auxiliary loss and regularisation plot of infoGAN training")

        # Saving file:
        filepath = os.path.join(self.log_path, "info_reg_loss_plot.png")
        plt.savefig(filepath, dpi=300, metadata=self.metadata)
        plt.close()

        # Generating BCE plot:
        fig, ax = plt.subplots()
        ax.plot(disc_class_loss)
        ax.plot(gen_class_loss)
        plt.legend(
            ("Discriminator Loss", "Generator Loss"),
            loc='upper right')
        plt.title("Discriminator and Generator loss during infoGAN training")

        # Saving file:
        filepath = os.path.join(self.log_path, "bce_loss_plot.png")
        plt.savefig(filepath, dpi=300, metadata=self.metadata)
        plt.close()

    def plot_generated_images(self, real_images: torch.Tensor) -> None:
        """Save plots of real images and the network's reconstruction of these images."""
        # Saving original set of images:
        original_filepath = os.path.join(self.log_path, "original_images.png")
        save_image(real_images, original_filepath, nrow=10)

        # Finding latent codes:
        self.model.discriminator.eval()
        self.model.aux_head.eval()
        features = self.model.discriminator(real_images)
        latent_codes = self.model.aux_head(features)

        # Generating fake images:
        fake_images = self.model.generator(latent_codes)
        generated_filepath = os.path.join(self.log_path, "generated_images.png")
        save_image(fake_images, generated_filepath, nrow=10)
