import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold

import preprocessing
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

    trainer = pl.Trainer(
        max_epochs=300,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
    )

    # Perform whole training loop
    trainer.fit(model, train_dataloader, val_dataloader)

    preds_batches = trainer.predict(model, val_dataloader)
    preds = torch.cat(preds_batches)  # type: ignore

    return preds


def cross_val_train(
    model_cls,
    model_args: dict,
    dataset: data.TensorDataset,
    y: np.ndarray | torch.Tensor,
    n_splits: int = 5,
    seed: int = 42,
) -> list[float]:

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.empty(y.shape[0])

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(y)), y)):
        model = model_cls(**model_args)
        preds_fold = train(model, dataset, train_idx, val_idx)
        preds[val_idx] = preds_fold.numpy()

    return preds.tolist()
