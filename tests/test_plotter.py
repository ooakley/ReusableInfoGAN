import os

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


def test_plotter(
        full_infogan: solver.InfoGANHandler, cell_image_sample: data_utils.FullDataset) -> None:
    # Setting up test:
    full_infogan.train_on_epoch(cell_image_sample, batch_size=100)

    # Testing plotter:
    infogan_plotter = plotter.InfoGANPlotter(full_infogan)
    infogan_plotter.plot_loss_history()
    infogan_plotter.plot_generated_images(cell_image_sample[0:100])
    infogan_plotter.plot_gradient_history()
    infogan_plotter.plot_image_pca(cell_image_sample[:])
