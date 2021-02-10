"""A collection of fixtures for test set-up."""

from typing import List
import pytest
import torch
import torch.nn as nn
import json
import numpy as np
from numpy.random import default_rng
import infogan.model as model
import infogan.solver as solver


@pytest.fixture
def config_dict() -> dict:
    with open("config/test_cfg.json") as f:
        cfg_dict = json.load(f)
    return cfg_dict


@pytest.fixture
def random_numpy_image_array() -> np.ndarray:
    rng = default_rng(0)
    array = rng.random((100, 1, 64, 64))
    array = array.astype(np.float32)
    return array


@pytest.fixture
def random_numpy_noise_array() -> np.ndarray:
    rng = default_rng(0)
    array = rng.standard_normal((100, 10))
    array = array.astype(np.float32)
    return array


@pytest.fixture
def initialised_generator(config_dict: dict) -> nn.Module:
    # Seeding random number generation:
    np.random.seed(config_dict["seed"])
    torch.manual_seed(config_dict["seed"])
    torch.set_deterministic(True)

    # Instantiating and initialising model:
    generator_instance = model.GeneratorNetwork(config_dict["zdim"], config_dict["generator"])
    generator_instance.linear.apply(model.relu_kaiming_init_weights)
    generator_instance.conv.apply(model.relu_kaiming_init_weights)
    return generator_instance


@pytest.fixture
def initialised_discriminator(config_dict: dict) -> nn.Module:
    # Seeding random number generation:
    np.random.seed(config_dict["seed"])
    torch.manual_seed(config_dict["seed"])
    torch.set_deterministic(True)

    # Instantiating and initialising model:
    discriminator_instance = model.DiscriminatorNetwork(config_dict["discriminator"])
    discriminator_instance.conv.apply(model.leaky_relu_kaiming_init_weights)
    return discriminator_instance


@pytest.fixture
def initialised_class_head(config_dict: dict) -> nn.Module:
    # Seeding random number generation:
    np.random.seed(config_dict["seed"])
    torch.manual_seed(config_dict["seed"])
    torch.set_deterministic(True)

    # Instantiating and initialising model:
    class_head_instance = model.ClassificationHead(config_dict["discriminator"])
    class_head_instance.linear.apply(model.final_linear_init_weights)
    return class_head_instance


@pytest.fixture
def initialised_aux_head(config_dict: dict) -> nn.Module:
    # Seeding random number generation:
    np.random.seed(config_dict["seed"])
    torch.manual_seed(config_dict["seed"])
    torch.set_deterministic(True)

    # Instantiating and initialising model:
    aux_head_instance = model.AuxiliaryHead(config_dict)
    aux_head_instance.linear.apply(model.leaky_relu_kaiming_init_weights)
    return aux_head_instance


@pytest.fixture
def full_infogan(config_dict: dict) -> solver.InfoGANHandler:
    infogan_instance = solver.InfoGANHandler(config_dict)
    return infogan_instance


def network_changed(network1: nn.Module, network2: nn.Module) -> bool:
    for param1, param2 in zip(network1.children(), network2.children()):
        assert type(param1) == type(param2)
        weight_different: List[bool] = []
        bias_different: List[bool] = []
        if type(param1) in {nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d}:
            weights1 = np.array(param1.weight.data.cpu())
            weights2 = np.array(param2.weight.data.cpu())
            assert np.all(np.equal(weights1, weights1))
            assert np.all(np.equal(weights2, weights2))
            weight_different.append(bool(np.all(np.not_equal(weights1, weights2))))
            if param1.bias is not None:
                assert param2.bias is not None
                bias1 = np.array(param1.bias.data.cpu())
                bias2 = np.array(param2.bias.data.cpu())
                assert np.all(np.equal(bias1, bias1))
                assert np.all(np.equal(bias2, bias2))
                bias_different.append(bool(np.all(np.not_equal(bias1, bias2))))
    return (all(weight_different) and all(bias_different))


def network_not_changed(network1: nn.Module, network2: nn.Module) -> bool:
    for param1, param2 in zip(network1.children(), network2.children()):
        assert type(param1) == type(param2)
        weight_same: List[bool] = []
        bias_same: List[bool] = []
        if type(param1) in {nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d}:
            weights1 = np.array(param1.weight.data.cpu())
            weights2 = np.array(param2.weight.data.cpu())
            assert np.all(np.equal(weights1, weights1))
            assert np.all(np.equal(weights2, weights2))
            weight_same.append(bool(np.all(np.equal(weights1, weights2))))
            if param1.bias is not None:
                assert param2.bias is not None
                bias1 = np.array(param1.bias.data.cpu())
                bias2 = np.array(param2.bias.data.cpu())
                assert np.all(np.equal(bias1, bias1))
                assert np.all(np.equal(bias2, bias2))
                bias_same.append(bool(np.all(np.equal(bias1, bias2))))
    return (all(weight_same) and all(bias_same))
