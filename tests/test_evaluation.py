"""Tests for evaluation utilities."""
import torch

from digit_classification.evaluation import evaluate_model
from digit_classification.model import DigitClassifier


class FakeDataModule:
    """
    Serves a single fixed batch as the test set, standing in for a real DigitDataModule so evaluation logic can be tested without a checkpoint
    file or the real MNIST download.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def test_dataloader(self):
        return [(self.images, self.labels)]


def test_evaluate_model_returns_report_with_all_classes():
    model = DigitClassifier()
    model.eval()

    images = torch.rand(6, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    datamodule = FakeDataModule(images, labels)

    report = evaluate_model(model=model, datamodule=datamodule)

    assert "0" in report
    assert "5" in report
    assert "8" in report
    assert "accuracy" in report
