"""Suite of unit tests for infoGAN training functionality."""
import copy
import math
import numpy as np
import torch
from .fixtures import (
    config_dict,
    random_numpy_image_array,
    random_numpy_noise_array,
    full_infogan,
    network_changed,
    network_not_changed
)
import infogan.solver as solver


def test_discriminator_train(
        full_infogan: solver.InfoGANHandler, random_numpy_image_array: np.ndarray,
        random_numpy_noise_array: np.ndarray
        ) -> None:
    infogan_to_update = copy.deepcopy(full_infogan)
    real_images = torch.Tensor(random_numpy_image_array)
    real_images.requires_grad = True
    noise_vector = torch.Tensor(random_numpy_noise_array)

    init_class_loss, init_disc_reg, _, _ = full_infogan.calculate_loss(real_images, noise_vector)
    initial_loss = init_class_loss + init_disc_reg
    assert math.isclose(init_class_loss, 0.301, abs_tol=0.5)

    # Performing update loop:
    class_loss, disc_reg, _, _ = infogan_to_update.calculate_loss(real_images, noise_vector)
    loss = class_loss + disc_reg
    infogan_to_update.discriminator_opt.zero_grad(set_to_none=True)
    loss.backward()
    infogan_to_update.discriminator_opt.step()
    train_class_loss, train_disc_reg, _, _ = \
        infogan_to_update.calculate_loss(real_images, noise_vector)
    train_loss = train_class_loss + train_disc_reg

    # Performing comparisons:
    assert train_loss < initial_loss
    assert network_changed(full_infogan.discriminator.conv, infogan_to_update.discriminator.conv)
    assert network_changed(full_infogan.class_head.linear, infogan_to_update.class_head.linear)
    assert network_not_changed(full_infogan.generator.conv, infogan_to_update.generator.conv)
    assert network_not_changed(full_infogan.aux_head.linear, infogan_to_update.aux_head.linear)


def test_generator_train(
        full_infogan: solver.InfoGANHandler, random_numpy_image_array: np.ndarray,
        random_numpy_noise_array: np.ndarray
        ) -> None:
    infogan_to_update = copy.deepcopy(full_infogan)
    real_images = torch.Tensor(random_numpy_image_array)
    real_images.requires_grad = True
    noise_vector = torch.Tensor(random_numpy_noise_array)

    _, _, init_class_loss, init_info_loss = full_infogan.calculate_loss(real_images, noise_vector)
    initial_loss = init_class_loss + init_info_loss
    assert math.isclose(init_class_loss, 0.301, abs_tol=0.5)

    # Performing update loop:
    _, _, class_loss, info_loss = infogan_to_update.calculate_loss(real_images, noise_vector)
    loss = class_loss + info_loss
    infogan_to_update.generator_opt.zero_grad(set_to_none=True)
    loss.backward()
    infogan_to_update.generator_opt.step()
    _, _, train_class_loss, train_info_loss = \
        infogan_to_update.calculate_loss(real_images, noise_vector)
    train_loss = train_class_loss + train_info_loss

    # Performing comparisons:
    assert train_loss < initial_loss
    assert network_not_changed(
            full_infogan.discriminator.conv, infogan_to_update.discriminator.conv)
    assert network_not_changed(full_infogan.class_head.linear, infogan_to_update.class_head.linear)
    assert network_changed(full_infogan.generator.conv, infogan_to_update.generator.conv)
    assert network_changed(full_infogan.aux_head.linear, infogan_to_update.aux_head.linear)
