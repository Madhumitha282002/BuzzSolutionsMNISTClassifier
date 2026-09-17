# Digit Classification

A command line tool that curates an imbalanced 3-class subset of MNIST(digits 0, 5, 8), trains a small CNN on it with PyTorch Lightning, and
supports evaluation and single-image inference.

## Setup

```bash
pip install -e ".[test]"
```

## Usage

```bash
digit-classification download-data --data-dir ./data

digit-classification train --data-dir ./data --output-dir ./output --epochs 20

digit-classification evaluate --checkpoint-path ./output/digit_classifier.ckpt --data-dir ./data

digit-classification predict --checkpoint-path ./output/digit_classifier.ckpt --input-path ./sample_digit.png
```

Run the tests with:

```bash
pytest
```

## Design decisions

**Reproducibility.** All random sampling uses `numpy.random.default_rng(seed)` rather than the global random state. `curate_indices` selects 3,500 images of "8", 1,200 of "0", and 300 of "5" from the MNIST training set, offsetting the seed by label so each class's sample is independent of the others. `split_indices` then reproducibly carves out 20% of that curated set for evaluation, and 20% of what remains for validation. Same seed in, same split out, every run.

**Label remapping.** The model only has 3 output units, so raw MNIST labels {0, 5, 8} are remapped to class indices {0, 1, 2} in `DigitDataset`.`LABELS` and `LABEL_TO_INDEX` in `data.py` are the single source of truth for that mapping so it's never duplicated as magic numbers elsewhere.

**Model architecture.** With roughly 5,000 total training images spread unevenly across three classes, a deep network would overfit almost immediately, especially on the 300-image "5" class. The model is two small convolutional blocks (16 then 32 filters) followed by two fully connected layers, with dropout after both the convolutional stack and the hidden dense layer. This keeps the parameter count low relative to the dataset size while still giving the model enough capacity to separate three fairly distinct digit shapes.

**Class imbalance.** The assignment doesn't ask for the imbalance to be corrected, only preserved and worked with, so no oversampling or class
weighting is applied. `evaluate`'s classification report (precision/recall/F1 per class) is the right lens here: it will make it obvious if the model is just learning to predict "8" most of the time, which plain accuracy would hide.

**Deviations from the provided stub, and why:**
- `training_step`, `validation_step`, and `predict_step` include a
  `batch_idx` parameter. The stub in the assignment omits it, but Lightning's
  `Trainer` calls these hooks with `(batch, batch_idx)`, so leaving it out
  would break training.
- The `predict` command's `--input-path` option was written as `--output-dir`
  in the provided CLI stub. That looks like a copy-paste typo in the
  assignment template, so it's corrected here to `--input-path`.

## Project structure

```
digit-classification/
├── README.md
├── pyproject.toml
├── src/digit_classification/
│   ├── __init__.py
│   ├── cli.py           # Typer commands: download-data, train, evaluate, predict
│   ├── data.py           # Curation, splitting, Dataset, DataModule
│   ├── model.py           # LightningModule (CNN)
│   └── evaluation.py     # Classification report generation
└── tests/
    ├── test_data.py
    ├── test_model.py
    └── test_evaluation.py
```

Data and model tests use small synthetic tensors rather than the real MNIST download, so the suite runs quickly and works offline.
