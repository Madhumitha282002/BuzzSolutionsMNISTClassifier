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

**Reproducibility.** I used `numpy.random.default_rng(seed)` for the random sampling instead of the global random state so that I can get the same results when I use the same seed. `curate_indices` selects 3,500 images of "8", 1,200 of "0", and 300 of "5" from the MNIST training set. I offset the seed for each label so that the sampling for one class doesn't affect the others. After that, `split_indices` uses the same approach to create the evaluation and validation splits. This means that running the process again with the same seed gives me the same splits.

**Label remapping.** Since the model is only classifying 0, 5, and 8, I remap the original MNIST labels `{0, 5, 8}` to `{0, 1, 2}` before passing them to the model. The `LABELS` and `LABEL_TO_INDEX` variables in `data.py` keep this mapping in one place instead of having the mapping repeated throughout the code.

**Model architecture.** I used a small CNN with two convolutional layers (16 and 32 filters), followed by two fully connected layers. I also added dropout to help with overfitting. Since the dataset is relatively small and the classes are unevenly distributed, especially with only 300 examples of "5", I wanted to keep the model fairly simple rather than using a much deeper network.

**Class imbalance.** I kept the class imbalance from the curated dataset because the assignment asks for it to be preserved. I didn't use oversampling or class weighting. Instead, I use precision, recall, and F1-score in the classification report so I can see how the model performs on each digit individually, rather than relying only on overall accuracy.

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
