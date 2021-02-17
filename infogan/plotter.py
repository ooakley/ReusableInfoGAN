"""Class for methods useful for ouputting metrics related to infoGAN performance."""
import os
from datetime import datetime
from typing import List, Iterator, Tuple


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
import torch
from torchvision.utils import save_image

from infogan import solver


class GradientTracker:

    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots()
        self.initialised: bool = False

    def update_gradient_plot(self, named_parameters: List[Tuple[str, bool, torch.Tensor]]) -> None:
        ave_grads = []
        layers = []
        for n, p_req_grad, p_grad in named_parameters:
            if(p_req_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p_grad)
        if not self.initialised:
            self.ax.plot(ave_grads, alpha=0.3, color="b")
            self.ax.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
            self.ax.set_xticks(range(0, len(ave_grads), 1))
            self.ax.set_xticklabels(layers, rotation="vertical")
            self.ax.set_xlim(xmin=0, xmax=len(ave_grads) - 1)
            self.ax.set_xlabel("Layers")
            self.ax.set_ylabel("Average gradient")
            self.ax.set_title("Gradient flow")
            self.ax.grid(True)
        if self.initialised:
            self.ax.plot(ave_grads, alpha=0.3, color="b")

    def save_gradient_plot(self, path: str) -> None:
        self.fig.tight_layout()
        self.fig.savefig(path, dpi=300)
        plt.close(self.fig)


class InfoGANPlotter:
    """Class to encapsulate methods for generate infoGAN metrics."""

    def __init__(self, model: solver.InfoGANHandler) -> None:
        """Initialise plotter with trained model."""
        self.model = model
        self.log_path = os.path.join(model.log_path, "plots")
        if not os.path.isdir(self.log_path):
            os.mkdir(self.log_path)
        self.loss_history = model.loss_history
        self.metadata = {
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
        fig.savefig(filepath, dpi=300, metadata=self.metadata)
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
        fig.savefig(filepath, dpi=300, metadata=self.metadata)
        plt.close()

    def plot_generated_images(self, real_images: torch.Tensor) -> None:
        """Save plots of real images and the network's reconstruction of these images."""
        # Saving original set of images:
        real_images = real_images.to(self.model.device)
        original_filepath = os.path.join(self.log_path, "original_images.png")
        save_image(real_images, original_filepath, nrow=10, normalize=True, range=(-1, 1))

        # Finding latent codes:
        self.model.discriminator.eval()
        self.model.aux_head.eval()
        features = self.model.discriminator(real_images)
        latent_codes = self.model.aux_head(features)
        noise_variable = torch.randn(latent_codes.shape[0], 1).to(self.model.device)
        full_codes = torch.cat((latent_codes, noise_variable), 1)
        self.model.discriminator.train()
        self.model.aux_head.train()

        # Generating fake images:
        fake_images = self.model.generator(full_codes)
        generated_filepath = os.path.join(self.log_path, "generated_images.png")
        save_image(fake_images, generated_filepath, nrow=10, normalize=True, range=(-1, 1))

    def plot_gradient_history(self, path: str = None) -> None:
        """Save plots of gradient history across different layers."""
        # Defining filepaths:
        if path is None:
            disc_filepath = os.path.join(self.log_path, "disc_grad.png")
            gen_filepath = os.path.join(self.log_path, "gen_grad.png")
        if path is not None:
            disc_filepath = os.path.join(path, "disc_grad.png")
            gen_filepath = os.path.join(path, "gen_grad.png")

        # Instantiating plotters:
        disc_tracker = GradientTracker()
        gen_tracker = GradientTracker()

        # Plotting gradients:
        for disc_params, gen_params in zip(
                self.model.discriminator_grad_history, self.model.generator_grad_history
                ):
            disc_tracker.update_gradient_plot(disc_params)
            gen_tracker.update_gradient_plot(gen_params)

        # Saving to file:
        disc_tracker.save_gradient_plot(disc_filepath)
        gen_tracker.save_gradient_plot(gen_filepath)

    def plot_image_pca(self, real_images: torch.Tensor, path: str = None) -> None:
        """
        Save scatterplot of images embedded into latent space.

        Derived from:
        https://www.kaggle.com/gaborvecsei/plants-t-sne
        """
        # Generating path:
        if path is None:
            filepath = os.path.join(self.log_path, "embeddings_pca_plot.png")
        if path is not None:
            filepath = os.path.join(path, "embeddings_pca_plot.png")

        # Generating embeddings:
        real_images = real_images.to(self.model.device)
        self.model.discriminator.eval()
        self.model.aux_head.eval()
        features = self.model.discriminator(real_images)
        latent_codes = self.model.aux_head(features)
        np_latent_codes = latent_codes.detach().cpu().numpy()
        embeddings = PCA(n_components=2).fit_transform(np_latent_codes)
        self.model.discriminator.train()
        self.model.aux_head.train()

        # Generating scatterplot:
        plot_images = (real_images[:, 0, :, :].detach().cpu().numpy() + 1) / 2
        fig, ax = plt.subplots(figsize=(60, 60))
        artists = []
        for i in range(plot_images.shape[0]):
            img_array = plot_images[i].squeeze()
            img = OffsetImage(img_array, zoom=1, cmap='gray')
            ab = AnnotationBbox(img,
                (embeddings[i, 0], embeddings[i, 1]),
                xycoords='data', frameon=False
            )
            artists.append(ax.add_artist(ab))
        
        # Tweaking plot:
        ax.update_datalim(embeddings)
        ax.autoscale()
        fig.set_facecolor("k")
        ax.set_facecolor("k")
        ax.spines['bottom'].set_color('w')
        ax.spines['left'].set_color('w')
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')

        # Saving to file:
        fig.tight_layout()
        fig.savefig(filepath, dpi=150, metadata=self.metadata)
        plt.close()
