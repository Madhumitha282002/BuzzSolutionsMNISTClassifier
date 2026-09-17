"""Tests for dataset curation and splitting logic."""

import numpy as np
import torch

from digit_classification.data import (
    CLASS_COUNTS,
    DigitDataset,
    LABEL_TO_INDEX,
    _select_indices,
    curate_indices,
    split_indices,
)


class FakeMNIST:
    """
    Generates synthetic targets with enough of each label to satisfy CLASS_COUNTS, so tests don't depend on downloading the real dataset.
    """

    def __init__(self):
        targets = []
        for label, count in CLASS_COUNTS.items():
            targets.extend([label] * (count + 50))
        for other_label in [1, 2, 3, 4, 6, 7, 9]:
            targets.extend([other_label] * 20)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.data = torch.randint(0, 255, (len(targets), 28, 28), dtype=torch.uint8)


def test_select_indices_returns_correct_count_and_label():
    targets = torch.tensor([0, 5, 8, 0, 5, 8, 0, 0, 5, 8])
    indices = _select_indices(targets, label=0, count=3, seed=1)
    assert len(indices) == 3
    assert all(targets[i] == 0 for i in indices)


def test_select_indices_is_reproducible():
    targets = torch.tensor([0, 5, 8] * 20)
    first = _select_indices(targets, label=5, count=5, seed=42)
    second = _select_indices(targets, label=5, count=5, seed=42)
    assert np.array_equal(first, second)


def test_curate_indices_has_expected_class_counts():
    mnist = FakeMNIST()
    curated = curate_indices(mnist, seed=42)
    assert len(curated) == sum(CLASS_COUNTS.values())

    curated_labels = mnist.targets[curated].tolist()
    for label, count in CLASS_COUNTS.items():
        assert curated_labels.count(label) == count


def test_curate_indices_is_reproducible():
    mnist = FakeMNIST()
    first = curate_indices(mnist, seed=42)
    second = curate_indices(mnist, seed=42)
    assert np.array_equal(first, second)


def test_split_indices_produces_correct_proportions():
    indices = np.arange(1000)
    train_idx, val_idx, test_idx = split_indices(indices, seed=42)

    assert len(test_idx) == 200
    assert len(train_idx) + len(val_idx) == 800
    assert len(val_idx) == 160

    all_indices = np.concatenate([train_idx, val_idx, test_idx])
    assert len(all_indices) == len(set(all_indices.tolist()))


def test_split_indices_is_reproducible():
    indices = np.arange(500)
    first = split_indices(indices, seed=7)
    second = split_indices(indices, seed=7)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_digit_dataset_remaps_labels_and_scales_images():
    images = torch.randint(0, 255, (4, 28, 28), dtype=torch.uint8)
    labels = torch.tensor([0, 5, 8, 0])

    dataset = DigitDataset(images, labels)

    assert len(dataset) == 4
    image, label = dataset[0]
    assert image.shape == (1, 28, 28)
    assert image.max() <= 1.0
    assert image.min() >= 0.0
    assert label.item() == LABEL_TO_INDEX[0]
