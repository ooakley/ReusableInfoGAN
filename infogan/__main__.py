import os
import json
import argparse
from typing import Any, List

import numpy as np

from . import data_utils, solver, plotter


def _parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_dict", type=str,
        help="which config file to use"
    )
    return parser.parse_args()


def generate_dataset(path_to_data: str) -> data_utils.FullDataset:
    """Generate torch dataset from path to folder containing numpy arrays."""
    numpy_files = os.listdir(path_to_data)
    numpy_dataset: List[np.ndarray] = []
    for f in numpy_files:
        path = os.path.join(path_to_data, f)
        data = np.expand_dims(np.load(path)[:, :, :, 1], 1).astype(np.float32)
        numpy_dataset.append(data)
    dataset = data_utils.FullDataset(np.concatenate(numpy_dataset, axis=0))
    return dataset


def main() -> None:
    """Run basic training loop."""
    # Parsing arguments and loading config file:
    args = _parse_args()
    config_path = os.path.join("config", args.config_dict + ".json")
    with open(config_path) as f:
        config_dict = json.load(f)

    # Loading dataset:
    print("Loading dataset...")
    numpy_data_path = os.path.join("data", "numpy_datasets", config_dict["dataset"])
    dataset = generate_dataset(numpy_data_path)

    # Instantiating handler:
    print("Initialising model...")
    handler = solver.InfoGANHandler(config_dict)

    # Training model:
    print("Training model...")
    for i in range(config_dict["epochs"]):
        handler.train_on_epoch(dataset, config_dict["batch_size"])

    # Running plotter:
    infogan_plotter = plotter.InfoGANPlotter(handler)
    infogan_plotter.plot_loss_history()
    infogan_plotter.plot_generated_images(dataset[0:100])


main()
