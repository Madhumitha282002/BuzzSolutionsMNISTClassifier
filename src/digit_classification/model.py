"""PyTorch Lightning model for classifying MNIST digits 0, 5, and 8."""

import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch import LightningModule
from torch import nn

NUM_CLASSES = 3


class DigitClassifier(LightningModule):
    """A small CNN for classifying digits 0, 5, and 8.
    """

    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, NUM_CLASSES),
        )

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASSES)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.conv_block(x)
            return self.classifier(features)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx: int = 0, dataloader_idx: int = 0) -> torch.Tensor:
        """Return the predicted probability of each digit label for the batch."""


        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = self(images)
        return F.softmax(logits, dim=1)
