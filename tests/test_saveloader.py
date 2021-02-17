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
from infogan.saveloader import SaveLoader

def test_save_logic(
        full_infogan: solver.InfoGANHandler, cell_image_sample: data_utils.FullDataset,
        random_numpy_noise_array: np.ndarray) -> None:
    # Setting up single round of training:
    noise_vector = torch.Tensor(random_numpy_noise_array)
    real_images = cell_image_sample[0:100]
    real_images.requires_grad = True

    # Training model:
    full_infogan.train_on_batch(real_images, noise_vector)

    # Saving model:
    SaveLoader.save(full_infogan)

    # Loading model:
    path = full_infogan.log_path
    load_infogan = SaveLoader.load(path)

    # Training both models one more time:
    full_infogan.train_on_batch(real_images, noise_vector)
    load_infogan.train_on_batch(real_images, noise_vector)
    
    # Asserting both models are entirely equal:
    assert network_not_changed(full_infogan.generator.linear, load_infogan.generator.linear)
    assert network_not_changed(full_infogan.generator.conv, load_infogan.generator.conv)
    assert network_not_changed(full_infogan.discriminator.conv, load_infogan.discriminator.conv)
    assert network_not_changed(full_infogan.class_head.linear, load_infogan.class_head.linear)
    assert network_not_changed(full_infogan.aux_head.linear, load_infogan.aux_head.linear)

    # Removing logs:
    os.remove(os.path.join(full_infogan.log_path, "model_state.pth"))
    os.remove(os.path.join(full_infogan.log_path, "cfg.json"))
    os.rmdir(full_infogan.log_path)
    os.remove(os.path.join(load_infogan.log_path, "cfg.json"))
    os.rmdir(load_infogan.log_path)
