from __future__ import annotations

import os
import json
import argparse
from typing import Any

import torch
import numpy as np

from . import data_utils, solver, plotter
from .saveloader import SaveLoader


def _parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dict", type=str, default=None,
        help="which config file to use"
    )
    parser.add_argument(
        "-t", "--train", action="store_true",
        help="whether to train network"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="whether to turn off plotting of network diagnostics - such as loss history"
    )
    parser.add_argument(
        "-i", "--infer", type=str, default=None,
        help="takes in path to dataset to generate latent space embeddings, saves to output/ dir"
    )
    parser.add_argument(
        "-l", "--load", type=str, default=None,
        help="takes in path to model log dir, instantiates model - ignores --config_dict arg"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=None,
        help="overwrites config epochs - only useful for resuming training after load"
    )
    return parser.parse_args()


def generate_dataset(path_to_data: str) -> dict[str, np.ndarray]:
    """Generate torch dataset from path to folder containing numpy arrays."""
    numpy_files = os.listdir(path_to_data)
    numpy_dataset: dict[str, np.ndarray] = {}
    for f in numpy_files:
        path = os.path.join(path_to_data, f)
        data = np.expand_dims(np.load(path)[:, :, :, 1], 1).astype(np.float32)
        scaled_data = (data * 2) - 1
        numpy_dataset[f] = scaled_data
    return numpy_dataset


def main() -> None:
    """Run basic training loop."""
    # Parsing arguments and loading config file:
    args = _parse_args()
    if args.config_dict is not None:
        config_path = os.path.join("config", args.config_dict + ".json")
        with open(config_path) as f:
            config_dict = json.load(f)

    # Setting device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiating handler:
    print("Initialising model...")
    if args.load is None:
        handler = solver.InfoGANHandler(config_dict, device=device)
    if args.load is not None:
        handler = SaveLoader.load(args.load)
        config_path = os.path.join(args.load, "cfg.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        if args.epochs is not None:
            config_dict["epochs"] = args.epochs

    # Loading dataset:
    print("Loading dataset...")
    numpy_data_path = os.path.join("data", "numpy_datasets", config_dict["dataset"])
    np_dataset = generate_dataset(numpy_data_path)
    dataset = data_utils.FullDataset(np.concatenate(np_dataset.values(), axis=0))

    # Training model:
    print("Training model...")
    if args.train:
        for i in range(config_dict["epochs"]):
            handler.train_on_epoch(dataset, config_dict["batch_size"])
            print(i + 1)

    # Running plotter:
    if not args.quiet:
        infogan_plotter = plotter.InfoGANPlotter(handler)
        infogan_plotter.plot_loss_history()
        infogan_plotter.plot_generated_images(dataset[0:100])
        infogan_plotter.plot_gradient_history()
        infogan_plotter.plot_image_pca(dataset[::500, :, :, :])

    # Saving model:
    SaveLoader.save(handler)

    # Generating latent space embeddings of dataset:
    if args.infer:
        embeddings: dict[str, np.ndarray] = {}
        for array_name in np_dataset:
            temp_dataset = data_utils.FullDataset(np_dataset[array_name])
            temp_dataloader = torch.utils.data.DataLoader(
                temp_dataset, batch_size=1024, shuffle=False, num_workers=4
            )

main()
