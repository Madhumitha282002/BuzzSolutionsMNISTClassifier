"""
Data curation and loading utilities for the digit classification project.

The problem statement asks for a deliberately imbalanced 3-class subset of MNIST (digits 0, 5, 8) with 3,500 / 1,200 / 300 images respectively. All
random selection here uses numpy's Generator API seeded explicitly, so the same seed always produces the same curated set and the same train/val/test
split.
"""

from typing import Tuple

import numpy as np
import torch
import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Tuple
from torchvision.datasets import MNIST

# The three digit classes this project cares about, in a fixed order.
# Model output index i always corresponds to LABELS[i].
LABELS = [0, 5, 8]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}

# How many images of each digit to pull out of the full MNIST training set.
CLASS_COUNTS = {
    8: 3500,
    0: 1200,
    5: 300,
}

SEED = 42
TEST_FRACTION = 0.2  # held out for evaluation
VAL_FRACTION = 0.2  # fraction of the *remaining* data used for validation


def download_mnist(data_dir: str) -> None:
    MNIST(root=data_dir, train=True, transform=None, download=True)


def _select_indices(targets: torch.Tensor, label: int, count: int, seed: int) -> np.ndarray:
    label_indices = np.where(targets.numpy() == label)[0]
    rng = np.random.default_rng(seed + label)
    return rng.choice(label_indices, size=count, replace=False)


def curate_indices(mnist_dataset: MNIST, seed: int = SEED) -> np.ndarray:
    """Build the imbalanced 5,000-image index set described in the assignment."""
    targets = mnist_dataset.targets
    selected = [_select_indices(targets, label, count, seed) for label, count in CLASS_COUNTS.items()]
    combined = np.concatenate(selected)
    # Shuffle so that batches later on aren't accidentally ordered by class.
    np.random.default_rng(seed).shuffle(combined)
    return combined


def split_indices(indices: np.ndarray, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split curated indices into train, validation, and test sets.
    20% of the curated data is held out for evaluation. Of the remaining 80%, a further 20% is used for validation during training. Both splits are reproducible given the same seed.
    """
    rng = np.random.default_rng(seed)
    shuffled = indices.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(round(n_total * TEST_FRACTION))
    test_indices = shuffled[:n_test]
    remaining = shuffled[n_test:]

    n_val = int(round(len(remaining) * VAL_FRACTION))
    val_indices = remaining[:n_val]
    train_indices = remaining[n_val:]

    return train_indices, val_indices, test_indices


class DigitDataset(Dataset):
    """
    Wraps a subset of MNIST images and labels for the 3-class problem.
    Images are scaled to [0, 1] and given a channel dimension. Labels are remapped from {0, 5, 8} to class indices {0, 1, 2} via LABEL_TO_INDEX
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images.float().unsqueeze(1) / 255.0  # (N, 1, 28, 28)
        self.labels = torch.tensor([LABEL_TO_INDEX[int(label)] for label in labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


class DigitDataModule(LightningDataModule):
    """Lightning data module that curates, splits, and serves the MNIST subset."""

    def __init__(self, data_dir: str, batch_size: int = 32, seed: int = SEED):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        download_mnist(self.data_dir)

    def setup(self, stage: str = None) -> None:
        mnist_dataset = MNIST(root=self.data_dir, train=True, transform=None, download=False)
        curated = curate_indices(mnist_dataset, seed=self.seed)
        train_idx, val_idx, test_idx = split_indices(curated, seed=self.seed)

        self.train_dataset = DigitDataset(mnist_dataset.data[train_idx], mnist_dataset.targets[train_idx])
        self.val_dataset = DigitDataset(mnist_dataset.data[val_idx], mnist_dataset.targets[val_idx])
        self.test_dataset = DigitDataset(mnist_dataset.data[test_idx], mnist_dataset.targets[test_idx])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=9)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=9)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
