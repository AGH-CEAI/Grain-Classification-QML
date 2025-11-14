import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold

import preprocessing


def get_dataloader(
    dataset: data.TensorDataset,
    indices: np.ndarray,
    shuffle: bool = False,
    batch_size: int = 32,
) -> data.DataLoader:

    if indices is not None:
        subset = data.Subset(dataset, list(indices))
        ds = subset
    else:
        ds = dataset

    return data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )


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
    preds = torch.cat(preds_batches)

    return preds


def cross_val_train(
    model: pl.LightningModule, X: np.ndarray, y: np.ndarray
) -> list[float]:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dataset = preprocessing.get_tensor_dataset(X, y)
    preds = np.empty(y.shape[0])

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):

        preds_fold = train(model, dataset, train_idx, val_idx)
        preds[val_idx] = preds_fold.numpy()

    return preds.tolist()
