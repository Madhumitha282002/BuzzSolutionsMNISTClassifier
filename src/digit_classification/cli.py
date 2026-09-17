"""Command line interface for the digit classification project."""

from pathlib import Path

import torch
import torchvision.transforms.functional as TF
import typer
from lightning.pytorch import Trainer, seed_everything
from PIL import Image

from digit_classification.data import LABELS, SEED, DigitDataModule, download_mnist
from digit_classification.evaluation import evaluate_model, load_model
from digit_classification.model import DigitClassifier

app = typer.Typer()


@app.command("download-data")
def download_data(
    data_dir: str = typer.Option(..., "--data-dir"),
) -> None:
    download_mnist(data_dir)
    typer.echo(f"MNIST training data downloaded to {data_dir}")


@app.command()
def train(
    data_dir: str = typer.Option(..., "--data-dir"),
    output_dir: str = typer.Option(..., "--output-dir"),
    epochs: int = typer.Option(20, "--epochs"),
    batch_size: int = typer.Option(32, "--batch-size"), 
    learning_rate: float = typer.Option(1e-3, "--learning-rate"), 
    seed: int = typer.Option(SEED, "--seed"),) -> None:
    seed_everything(seed, workers=True)

    datamodule = DigitDataModule(data_dir=data_dir, batch_size=batch_size, seed=seed)
    model = DigitClassifier(learning_rate=learning_rate)

    trainer = Trainer(
        default_root_dir=output_dir,
        accelerator="cpu",
        max_epochs=epochs,
        deterministic=True,
    )
    trainer.fit(model, datamodule=datamodule)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path / "digit_classifier.ckpt"
    trainer.save_checkpoint(str(checkpoint_path))
    typer.echo(f"Model checkpoint saved to {checkpoint_path}")


@app.command()
def evaluate(
    checkpoint_path: str = typer.Option(..., "--checkpoint-path"),
    data_dir: str = typer.Option(..., "--data-dir"),
    seed: int = typer.Option(SEED, "--seed"),
) -> None:
    """Evaluate a trained model on the test set and print a classification report."""
    report = evaluate_model(checkpoint_path=checkpoint_path, data_dir=data_dir, seed=seed)
    typer.echo(report)


@app.command()
def predict(
    checkpoint_path: str = typer.Option(..., "--checkpoint-path"), 
    input_path: str = typer.Option(..., "--input-path"),) -> None:
    """Predict the digit label of a single image."""
    model = load_model(checkpoint_path)

    image = Image.open(input_path).convert("L").resize((28, 28))
    tensor = TF.to_tensor(image).unsqueeze(0)  # (1, 1, 28, 28), already scaled to [0, 1]

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    for label, probability in zip(LABELS, probabilities.tolist()):
        typer.echo(f"Digit {label}: {probability:.4f}")

    predicted_label = LABELS[int(torch.argmax(probabilities))]
    typer.echo(f"Predicted digit: {predicted_label}")


if __name__ == "__main__":
    app()
