from typing import Tuple
import mlflow
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.svm import SVC
import torch
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from logging_utils import (
    EpochMetricsTracker,
    log_metrics,
    start_child_hp_run,
)
from utils import get_dataloader


def train(
    model: pl.LightningModule,
    dataset: data.TensorDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
):
    # Create subsets for training and validation
    train_dataloader = get_dataloader(dataset, train_idx, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(dataset, val_idx, batch_size=32, shuffle=False)  #

    callback = EpochMetricsTracker()
    trainer = pl.Trainer(
        max_epochs=300,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=[callback],
        logger=False,
    )

    # Perform whole training loop
    trainer.fit(model, train_dataloader, val_dataloader)

    val_acc = trainer.callback_metrics["val_acc"].item()
    val_f1 = trainer.callback_metrics["val_f1"].item()
    metrics = {"accuracy": val_acc, "f1": val_f1}

    # Predict on validation and get metrics
    preds_batches = trainer.predict(model, val_dataloader)
    preds = torch.cat(preds_batches)  # type: ignore

    return preds, metrics


def cross_val_train(
    model_cls,
    model_args: dict,
    dataset: data.TensorDataset,
    y: np.ndarray | torch.Tensor,
    n_splits: int = 5,
    seed: int = 42,
) -> Tuple[list[float], dict]:

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.empty(y.shape[0])
    fold_metrics = {"accuracy": [], "f1": []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(y)), y)):
        with start_child_hp_run(f"Fold {fold}"):

            # Create model instance and train it
            model = model_cls(**model_args)
            preds_fold, metrics = train(model, dataset, train_idx, val_idx)

            # Get fold metrics
            preds[val_idx] = preds_fold.numpy()
            fold_metrics["accuracy"].append(metrics["accuracy"])
            fold_metrics["f1"].append(metrics["f1"])

    return preds.tolist(), fold_metrics


def cross_val_svm(
    model_args: dict,
    X,
    y,
    n_splits: int = 5,
    seed: int = 42,
):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    preds = np.empty(y.shape[0])
    fold_metrics = {"accuracy": [], "f1": []}  # store results for each fold

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        with start_child_hp_run(f"Fold {fold}"):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = SVC(**model_args)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            metrics = {
                "fold": fold,
                "final_val_acc": accuracy_score(y_test, y_pred),
                "precision": precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
                "final_val_f1": f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                ),
            }
            log_metrics(metrics)
            fold_metrics["accuracy"].append(metrics["final_val_acc"])
            fold_metrics["f1"].append(metrics["final_val_f1"])
            preds[test_idx] = y_pred

    return preds.tolist(), fold_metrics
