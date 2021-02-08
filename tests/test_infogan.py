"""Suite of tests for infoGAN functionality."""
import math
import copy
from typing import List
import pytest
import torch
import torch.nn as nn
import json
import numpy as np
# import numpy.typing as npt
from numpy.random import default_rng
import infogan.data_utils as data_utils
import infogan.model as model


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
    array = rng.random((100, 10))
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
    generator_instance.conv.apply(model.relu_kaiming_init_weights)
    generator_instance.linear.apply(model.relu_kaiming_init_weights)
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


def network_changed(network1: nn.Module, network2: nn.Module) -> bool:
    for param1, param2 in zip(network1.children(), network2.children()):
        assert type(param1) == type(param2)
        weight_different: List[bool] = []
        bias_different: List[bool] = []
        if type(param1) in {nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d}:
            weights1 = np.array(param1.weight.data.cpu())
            weights2 = np.array(param2.weight.data.cpu())
            bias1 = np.array(param1.bias.data.cpu())
            bias2 = np.array(param2.bias.data.cpu())
            assert np.all(np.equal(weights1, weights1))
            assert np.all(np.equal(weights2, weights2))
            weight_different.append(bool(np.all(np.not_equal(weights1, weights2))))
            bias_different.append(bool(np.all(np.not_equal(bias1, bias2))))
    return (all(weight_different) and all(bias_different))


def test_dataset_load(random_numpy_image_array: np.ndarray) -> None:
    tensor_dataset = data_utils.FullDataset(random_numpy_image_array)
    assert type(tensor_dataset[:]) == torch.Tensor
    assert len(tensor_dataset) == random_numpy_image_array.shape[0]
    assert tensor_dataset[:].size() == random_numpy_image_array.shape
    assert (tensor_dataset[:].cpu().numpy() == random_numpy_image_array).any()
    assert tensor_dataset[:].cpu().numpy().dtype == np.float32


def test_generator_instantiation_from_config(config_dict: dict) -> None:
    generator = model.GeneratorNetwork(config_dict["zdim"], config_dict["generator"])
    # initial dense layer parameters = 20,480 + 2048 (bias)
    # convolution parameters = 99,088 + 129 (bias)
    # batch norm parameters = 256
    # total parameters = 122,001
    parameter_number = sum(p.numel() for p in generator.parameters())
    assert parameter_number == 122001, str(parameter_number)


def test_generator_relu_kaiming_initialisation(config_dict: dict) -> None:
    generator = model.GeneratorNetwork(config_dict["zdim"], config_dict["generator"])
    init_generator = copy.deepcopy(generator)
    init_generator.conv.apply(model.relu_kaiming_init_weights)
    init_generator.linear.apply(model.relu_kaiming_init_weights)

    # Asserting that number of found modules is as expected:
    assert sum(1 for _ in generator.children()) == 2
    assert sum(1 for _ in generator.linear.children()) == 1
    assert sum(1 for _ in generator.conv.children()) == 14

    # Asserting initialisation changes weights of linear network:
    for m, init_m in zip(generator.linear.children(), init_generator.linear.children()):
        assert type(m) == type(init_m)
        if type(m) in {nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d}:
            orig_weights = np.array(m.weight.data.cpu())
            init_weights = np.array(init_m.weight.data.cpu())
            orig_bias = np.array(m.bias.data.cpu())
            init_bias = np.array(init_m.bias.data.cpu())
            assert np.all(np.equal(orig_weights, orig_weights))
            assert np.all(np.not_equal(orig_weights, init_weights))
            assert np.all(np.not_equal(orig_bias, init_bias))

        assert type(m) != nn.BatchNorm2d

    # Asserting initialisation changes weights of convolutional network:
    for m, init_m in zip(generator.conv.children(), init_generator.conv.children()):
        assert type(m) == type(init_m)
        if type(m) in {nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d}:
            orig_weights = np.array(m.weight.data.cpu())
            init_weights = np.array(init_m.weight.data.cpu())
            orig_bias = np.array(m.bias.data.cpu())
            init_bias = np.array(init_m.bias.data.cpu())
            assert np.all(np.equal(orig_weights, orig_weights))
            assert np.all(np.not_equal(orig_weights, init_weights))
            assert np.all(np.not_equal(orig_bias, init_bias))

        if type(m) == nn.BatchNorm2d:
            assert np.var(np.array(m.weight.data.cpu())) == 0.0
            assert np.var(np.array(m.bias.data.cpu())) == 0.0
            init_std_in_weight = np.std(np.array(init_m.weight.data.cpu()))
            init_std_in_bias = np.std(np.array(init_m.bias.data.cpu()))
            assert math.isclose(init_std_in_weight, 0.02, abs_tol=0.008)
            assert math.isclose(init_std_in_bias, 0.02, abs_tol=0.008)


def test_generator_forward_pass(random_numpy_noise_array: np.ndarray,
                                initialised_generator: nn.Module
                                ) -> None:
    output = initialised_generator(torch.from_numpy(random_numpy_noise_array))
    assert output.shape == (100, 1, 64, 64)

    upper_quartile_out = np.quantile(output.detach().cpu().numpy(), 0.75)
    lower_quartile_out = np.quantile(output.detach().cpu().numpy(), 0.25)
    mean_out = np.mean(output.detach().cpu().numpy())
    assert math.isclose(mean_out, 0.5, abs_tol=0.1)
    assert lower_quartile_out < 0.4
    assert upper_quartile_out > 0.4

    # TODO: Work out meaningful assertsions to make here - first attempt:
    # 0.67448 is the analytical upper quartile of an N~(1, 0) distribution - which is what we expect
    # the output of the BatchNorm & final conv layer to share quantile stats with. The conv layer is
    # followed by a sigmoid layer, and so we expect the quartiles to be sig(+-0.67448) = 0.66250,
    # 0.33749.
    # assert math.isclose(upper_quartile_out, 0.66250, abs_tol=0.05)
    # assert math.isclose(lower_quartile_out, 0.33749, abs_tol=0.05)
    # assert mean_out > 0


def test_discriminator_instantiation_from_config(config_dict: dict) -> None:
    discriminator = model.DiscriminatorNetwork(config_dict["discriminator"])
    # convolution parameters = 99,088 + 256 (bias)
    # batch norm parameters = 256
    # total parameters = 99,473
    parameter_number = sum(p.numel() for p in discriminator.parameters())
    assert parameter_number == 99856, str(parameter_number)


def test_discriminator_leaky_relu_kaiming_initialisation(config_dict: dict) -> None:
    discriminator = model.DiscriminatorNetwork(config_dict["discriminator"])
    init_discriminator = copy.deepcopy(discriminator)
    init_discriminator.conv.apply(model.leaky_relu_kaiming_init_weights)

    # Asserting that number of found modules is as expected:
    assert sum(1 for _ in discriminator.children()) == 1
    assert sum(1 for _ in discriminator.conv.children()) == 15

    # Asserting initialisation changes weights of convolutional network:
    for m, init_m in zip(discriminator.conv.children(), init_discriminator.conv.children()):
        assert type(m) == type(init_m)
        if type(m) in {nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d}:
            orig_weights = np.array(m.weight.data.cpu())
            init_weights = np.array(init_m.weight.data.cpu())
            orig_bias = np.array(m.bias.data.cpu())
            init_bias = np.array(init_m.bias.data.cpu())
            assert np.any(np.equal(orig_weights, orig_weights))
            assert np.any(np.not_equal(orig_weights, init_weights))
            assert np.any(np.not_equal(orig_bias, init_bias))

        if type(m) == nn.BatchNorm2d:
            assert np.var(np.array(m.weight.data.cpu())) == 0.0
            assert np.var(np.array(m.bias.data.cpu())) == 0.0
            init_std_in_weight = np.std(np.array(init_m.weight.data.cpu()))
            init_std_in_bias = np.std(np.array(init_m.bias.data.cpu()))
            assert math.isclose(init_std_in_weight, 0.02, abs_tol=0.01)
            assert math.isclose(init_std_in_bias, 0.02, abs_tol=0.01)


def test_discriminator_forward_pass(random_numpy_image_array: np.ndarray,
                                    initialised_discriminator: nn.Module
                                    ) -> None:
    output = initialised_discriminator(torch.from_numpy(random_numpy_image_array))
    assert output.shape == (100, 2048)
    upper_quartile_out = np.quantile(output.detach().cpu().numpy(), 0.75)
    lower_quartile_out = np.quantile(output.detach().cpu().numpy(), 0.25)
    mean_out = np.mean(output.detach().cpu().numpy())
    assert mean_out > 0
    assert lower_quartile_out < 0
    assert upper_quartile_out > 0.3

    # TODO: Work out meaningful assertsions to make here - first attempt:
    # 0.67448 is the analytical upper quartile of an N~(1, 0) distribution which is what we expect
    # the output of the BatchNorm layer to be. Additionally, as BatchNorm is followed by LeakyRelu,
    # we expect the lower quartile to be scaled by alpha (here 0.1).
    # assert math.isclose(upper_quartile_out, 0.67448, abs_tol=0.05)
    # assert math.isclose(lower_quartile_out, -0.067448, abs_tol=0.005)
    # assert mean_out > 0


def test_head_initialisation(config_dict: dict) -> None:
    auxiliary_head = model.AuxiliaryHead(config_dict)
    init_auxiliary_head = copy.deepcopy(auxiliary_head)
    init_auxiliary_head.linear.apply(model.relu_kaiming_init_weights)

    class_head = model.ClassificationHead(config_dict["discriminator"])
    init_class_head = copy.deepcopy(class_head)
    init_class_head.linear.apply(model.leaky_relu_kaiming_init_weights)

    assert network_changed(auxiliary_head.linear, init_auxiliary_head.linear)
    assert network_changed(class_head.linear, init_class_head.linear)


def test_discriminator_class_head_integration(random_numpy_image_array: np.ndarray,
                                              initialised_discriminator: nn.Module,
                                              config_dict: dict
                                              ) -> None:
    discriminator_output = initialised_discriminator(torch.from_numpy(random_numpy_image_array))
    class_head = model.ClassificationHead(config_dict["discriminator"])
    with torch.no_grad():
        probability_logit = class_head(discriminator_output)
        assert probability_logit is not None
        assert probability_logit.size() == (100, 1)
        assert math.isclose(probability_logit.mean().item(), 0, abs_tol=0.1)


def test_discriminator_auxiliary_head_integration(random_numpy_image_array: np.ndarray,
                                                  initialised_discriminator: nn.Module,
                                                  config_dict: dict
                                                  ) -> None:
    discriminator_output = initialised_discriminator(torch.from_numpy(random_numpy_image_array))
    auxiliary_head = model.AuxiliaryHead(config_dict)
    with torch.no_grad():
        predicted_codes = auxiliary_head(discriminator_output)
        assert predicted_codes is not None
        assert predicted_codes.size() == (100, 10)
        assert math.isclose(predicted_codes.mean().item(), 0, abs_tol=0.1)


def test_full_network_integration(random_numpy_noise_array: np.ndarray) -> None:
    pass
