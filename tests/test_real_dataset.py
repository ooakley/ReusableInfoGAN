import os
import math
import copy
import itertools
from typing import Iterator, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

from .fixtures import (
    config_dict,
    random_numpy_image_array,
    random_numpy_noise_array,
    full_infogan,
    network_changed,
    network_not_changed,
    cell_image_sample
)
import infogan.solver as solver
import infogan.data_utils as data_utils
import infogan.plotter as plotter


def plot_grad_flow(
        named_parameters: Iterator[Tuple[str, torch.Tensor]],
        path: str
        ) -> None:
    """
    Creds to Roshan Rane:
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/6
    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads) - 1)
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def test_initial_loss_values(
        full_infogan: solver.InfoGANHandler, cell_image_sample: data_utils.FullDataset,
        random_numpy_noise_array: np.ndarray
        ) -> None:
    # Removing logs:
    os.remove(os.path.join(full_infogan.log_path, "cfg.json"))
    os.rmdir(full_infogan.log_path)

    # Setting up images:
    real_images = cell_image_sample[0:100]
    real_images.requires_grad = True
    noise_vector = torch.Tensor(random_numpy_noise_array)
    fake_images = full_infogan.generator(noise_vector)

    # Examining init values for loss:
    init_class_loss, init_disc_reg = \
            full_infogan.calculate_discriminator_loss(real_images, fake_images)
    initial_loss = init_class_loss + init_disc_reg
    assert math.isclose(init_class_loss, 0.6931, abs_tol=0.25)


def test_batch_train(
        full_infogan: solver.InfoGANHandler, cell_image_sample: data_utils.FullDataset,
        random_numpy_noise_array: np.ndarray
        ) -> None:
    # Removing logs:
    os.remove(os.path.join(full_infogan.log_path, "cfg.json"))
    os.rmdir(full_infogan.log_path)

    # Setting up test:
    infogan_to_update = copy.deepcopy(full_infogan)
    noise_vector = torch.Tensor(random_numpy_noise_array)
    real_images = cell_image_sample[0:100]
    real_images.requires_grad = True

    # Finding initial loss:
    fake_images = full_infogan.generator(noise_vector)
    _, init_disc_reg = \
        full_infogan.calculate_discriminator_loss(real_images, fake_images)
    _, init_info_loss = \
        full_infogan.calculate_generator_loss(fake_images, noise_vector)

    # Training:
    infogan_to_update.train_on_batch(real_images, noise_vector)

    # Finding loss after one update:
    fake_images = infogan_to_update.generator(noise_vector)
    _, train_disc_reg = \
        infogan_to_update.calculate_discriminator_loss(real_images, fake_images)
    _, train_info_loss = \
        infogan_to_update.calculate_generator_loss(fake_images, noise_vector)

    # Plotting gradient flows:
    path = os.path.join("data", "tests")
    if not os.path.exists(path):
        os.mkdir(path)
    infogan_plotter = plotter.InfoGANPlotter(infogan_to_update)
    infogan_plotter.plot_gradient_history(path)

    # Performing comparisons:
    assert train_disc_reg < init_disc_reg
    assert train_info_loss < init_info_loss
    assert network_changed(full_infogan.discriminator.conv, infogan_to_update.discriminator.conv)
    assert network_changed(full_infogan.class_head.linear, infogan_to_update.class_head.linear)
    assert network_changed(full_infogan.generator.conv, infogan_to_update.generator.conv)
    assert network_changed(full_infogan.aux_head.linear, infogan_to_update.aux_head.linear)
