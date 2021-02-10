"""Classes for data loading and manipulation."""
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset


class FullDataset(Dataset):
    """Basic class encapulating dataset functions, primarily for use with Dataloader."""

    def __init__(self, numpy_array: np.ndarray) -> None:
        """Initialise with numpy array."""
        self.data = numpy_array

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.data.shape[0]

    def __getitem__(self, idx: Any) -> torch.Tensor:
        """Return items from dataset as torch tensors."""
        data_selection = self.data[idx]
        tensor = torch.from_numpy(data_selection)
        return tensor
