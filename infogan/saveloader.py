"""Contains class for handling save/load logic of the model."""
import os
import json
from typing import Union

import torch

from . import solver


class SaveLoader:

    @staticmethod
    def save(handler: solver.InfoGANHandler) -> None:
        # Collecting state variables:
        log_path = handler.log_path
        network_states = {
            "generator": handler.generator.state_dict(),
            "discriminator": handler.discriminator.state_dict(),
            "class_head": handler.class_head.state_dict(),
            "aux_head": handler.aux_head.state_dict()
        }
        opt_states = {
            "generator_opt": handler.generator_opt.state_dict(),
            "discriminator_opt": handler.discriminator_opt.state_dict()
        }
        training_history = {
            "loss_history": handler.loss_history,
            "epochs_performed": handler.epochs_performed,
            "discriminator_grad_history": handler.discriminator_grad_history,
            "generator_grad_history": handler.generator_grad_history
        }

        # Saving variables to path:
        state_dict = {
            "log_path": log_path,
            "network_states": network_states,
            "opt_states": opt_states,
            "training_history": training_history
        }
        torch.save(state_dict, os.path.join(log_path, "model_state.pth"))

    @staticmethod
    def load(path: str, device: Union[str, torch.device] = "cpu") -> solver.InfoGANHandler:
        # Load config dict:
        with open(os.path.join(path, "cfg.json")) as f:
            cfg_dict = json.load(f)
 
        # Load .pth file:
        state_dict = torch.load(os.path.join(path, "model_state.pth"))

        # Instantiate model:
        handler = solver.InfoGANHandler(cfg_dict, device=device)

        # Set weights:
        network_states = state_dict["network_states"]
        handler.generator.load_state_dict(network_states["generator"])
        handler.discriminator.load_state_dict(network_states["discriminator"])
        handler.class_head.load_state_dict(network_states["class_head"])
        handler.aux_head.load_state_dict(network_states["aux_head"])

        # Set optimiser vars:
        opt_states = state_dict["opt_states"]
        handler.generator_opt.load_state_dict(opt_states["generator_opt"])
        handler.discriminator_opt.load_state_dict(opt_states["discriminator_opt"])

        # Set training history:
        training_history = state_dict["training_history"]
        handler.loss_history = training_history["loss_history"]
        handler.epochs_performed = training_history["epochs_performed"]
        handler.discriminator_grad_history = training_history["discriminator_grad_history"]
        handler.generator_grad_history = training_history["generator_grad_history"]

        return handler
