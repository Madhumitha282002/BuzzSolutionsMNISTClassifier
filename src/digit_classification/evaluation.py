"""Evaluation utilities: run a trained model on the held-out test set."""

import torch
from sklearn.metrics import classification_report

from digit_classification.data import LABELS, DigitDataModule
from digit_classification.model import DigitClassifier


def load_model(checkpoint_path: str) -> DigitClassifier:
    model = DigitClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def evaluate_model(checkpoint_path: str = None, data_dir: str = None, seed: int = 42, model: DigitClassifier = None, datamodule: DigitDataModule = None,) -> str:
    if model is None:
        model = load_model(checkpoint_path)

    if datamodule is None:
        datamodule = DigitDataModule(data_dir=data_dir, seed=seed)
        datamodule.setup()

    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in datamodule.test_dataloader():
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    target_names = [str(label) for label in LABELS]
    return classification_report(all_labels, all_predictions, target_names=target_names, zero_division=0)
