"""Suite of unit tests for infoGAN training functionality."""
import os
import copy
import math
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .fixtures import (
    config_dict,
    random_numpy_image_array,
    random_numpy_noise_array,
    full_infogan,
    network_changed,
    network_not_changed
)
import infogan.solver as solver


def test_discriminator_regularisation_decrease(
        full_infogan: solver.InfoGANHandler, random_numpy_image_array: np.ndarray,
        random_numpy_noise_array: np.ndarray
        ) -> None:
    # Removing logs:
    os.remove(os.path.join(full_infogan.log_path, "cfg.json"))
    os.rmdir(full_infogan.log_path)

    # Setting up real and fake images:
    infogan_to_update = copy.deepcopy(full_infogan)
    real_images = torch.Tensor(random_numpy_image_array)
    real_images.requires_grad = True
    noise_vector = torch.Tensor(random_numpy_noise_array)
    fake_images = full_infogan.generator(noise_vector)

    # Calculating initial loss:
    initial_class_loss, init_disc_reg = \
            full_infogan.calculate_discriminator_loss(real_images, fake_images)
    initial_loss = init_disc_reg

    # Performing update loop:
    class_loss, disc_reg = \
        infogan_to_update.calculate_discriminator_loss(real_images, fake_images)
    loss = class_loss + disc_reg
    infogan_to_update.discriminator_opt.zero_grad(set_to_none=True)
    loss.backward()
    infogan_to_update.discriminator_opt.step()
    _, train_disc_reg = \
        infogan_to_update.calculate_discriminator_loss(real_images, fake_images)
    train_loss = train_disc_reg

    # Performing comparisons:
    assert train_disc_reg < init_disc_reg
    assert network_changed(full_infogan.discriminator.conv, infogan_to_update.discriminator.conv)
    assert network_changed(full_infogan.class_head.linear, infogan_to_update.class_head.linear)
    assert network_not_changed(full_infogan.generator.conv, infogan_to_update.generator.conv)
    assert network_not_changed(full_infogan.aux_head.linear, infogan_to_update.aux_head.linear)


def test_discriminator_train(
        full_infogan: solver.InfoGANHandler, random_numpy_image_array: np.ndarray,
        random_numpy_noise_array: np.ndarray
        ) -> None:
    # Removing logs:
    os.remove(os.path.join(full_infogan.log_path, "cfg.json"))
    os.rmdir(full_infogan.log_path)

    # Setting up real and fake images:
    infogan_to_update = copy.deepcopy(full_infogan)
    real_images = torch.Tensor(random_numpy_image_array)
    real_images.requires_grad = True
    noise_vector = torch.Tensor(random_numpy_noise_array)
    fake_images = full_infogan.generator(noise_vector)

    # Calculating initial loss:
    init_class_loss, init_disc_reg = \
            full_infogan.calculate_discriminator_loss(real_images, fake_images)
    initial_loss = init_class_loss + init_disc_reg
    assert math.isclose(init_class_loss, 0.693, abs_tol=0.4)

    # Performing update loop:
    class_loss, disc_reg = \
        infogan_to_update.calculate_discriminator_loss(real_images, fake_images)
    loss = class_loss + disc_reg
    infogan_to_update.discriminator_opt.zero_grad(set_to_none=True)
    loss.backward()
    infogan_to_update.discriminator_opt.step()
    train_class_loss, train_disc_reg = \
        infogan_to_update.calculate_discriminator_loss(real_images, fake_images)
    train_loss = train_class_loss + train_disc_reg

    # Performing comparisons:
    assert train_class_loss < init_class_loss
    assert network_changed(full_infogan.discriminator.conv, infogan_to_update.discriminator.conv)
    assert network_changed(full_infogan.class_head.linear, infogan_to_update.class_head.linear)
    assert network_not_changed(full_infogan.generator.conv, infogan_to_update.generator.conv)
    assert network_not_changed(full_infogan.aux_head.linear, infogan_to_update.aux_head.linear)


def test_generator_train(
        full_infogan: solver.InfoGANHandler, random_numpy_image_array: np.ndarray,
        random_numpy_noise_array: np.ndarray
        ) -> None:
    # Removing logs:
    os.remove(os.path.join(full_infogan.log_path, "cfg.json"))
    os.rmdir(full_infogan.log_path)

    # Setting up test:
    infogan_to_update = copy.deepcopy(full_infogan)
    noise_vector = torch.Tensor(random_numpy_noise_array)
    fake_images = infogan_to_update.generator(noise_vector)

    init_class_loss, init_info_loss = \
        full_infogan.calculate_generator_loss(fake_images, noise_vector)
    initial_loss = init_class_loss + init_info_loss
    assert math.isclose(init_class_loss, 0.693, abs_tol=0.4)

    # Performing update loop:
    class_loss, info_loss = \
        infogan_to_update.calculate_generator_loss(fake_images, noise_vector)
    loss = class_loss + info_loss
    infogan_to_update.generator_opt.zero_grad(set_to_none=True)
    loss.backward()
    infogan_to_update.generator_opt.step()
    train_class_loss, train_info_loss = \
        infogan_to_update.calculate_generator_loss(fake_images, noise_vector)
    train_loss = train_class_loss + train_info_loss

    # Performing comparisons:
    assert train_loss < initial_loss
    assert network_not_changed(
            full_infogan.discriminator.conv, infogan_to_update.discriminator.conv)
    assert network_not_changed(full_infogan.class_head.linear, infogan_to_update.class_head.linear)
    assert network_changed(full_infogan.generator.conv, infogan_to_update.generator.conv)
    assert network_changed(full_infogan.aux_head.linear, infogan_to_update.aux_head.linear)


def test_batch_train(
        full_infogan: solver.InfoGANHandler, random_numpy_image_array: np.ndarray,
        random_numpy_noise_array: np.ndarray
        ) -> None:
    # Removing logs:
    os.remove(os.path.join(full_infogan.log_path, "cfg.json"))
    os.rmdir(full_infogan.log_path)

    # Setting up test:
    infogan_to_update = copy.deepcopy(full_infogan)
    noise_vector = torch.Tensor(random_numpy_noise_array)
    real_images = torch.Tensor(random_numpy_image_array)
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

    # Performing comparisons:
    assert train_disc_reg < init_disc_reg
    assert train_info_loss < init_info_loss
    assert network_changed(full_infogan.discriminator.conv, infogan_to_update.discriminator.conv)
    assert network_changed(full_infogan.class_head.linear, infogan_to_update.class_head.linear)
    assert network_changed(full_infogan.generator.conv, infogan_to_update.generator.conv)
    assert network_changed(full_infogan.aux_head.linear, infogan_to_update.aux_head.linear)
