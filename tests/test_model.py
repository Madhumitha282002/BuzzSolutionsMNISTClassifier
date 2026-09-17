"""Tests for the DigitClassifier model."""

import torch

from digit_classification.model import NUM_CLASSES, DigitClassifier


def test_forward_output_shape():
    model = DigitClassifier()
    batch = torch.rand(8, 1, 28, 28)
    logits = model(batch)
    assert logits.shape == (8, NUM_CLASSES)


def test_predict_step_returns_valid_probabilities():
    model = DigitClassifier()
    batch = (torch.rand(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))
    probabilities = model.predict_step(batch, batch_idx=0)

    assert probabilities.shape == (4, NUM_CLASSES)
    row_sums = probabilities.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(4), atol=1e-5)


def test_training_step_returns_scalar_loss():
    model = DigitClassifier()
    batch = (torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 2, 0]))
    loss = model.training_step(batch, batch_idx=0)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_validation_step_returns_scalar_loss():
    model = DigitClassifier()
    batch = (torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 2, 0]))
    loss = model.validation_step(batch, batch_idx=0)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_configure_optimizers_returns_adam():
    model = DigitClassifier(learning_rate=5e-4)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 5e-4
