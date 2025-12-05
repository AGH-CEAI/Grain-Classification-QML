import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch.utils.data as data

from data.load_data import get_excel_data
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


def check_cross_val_split_sizes(n_splits=5, seed=42):
    # Load data
    df = get_excel_data()

    # Preprocessing
    x, y = preprocessing.preprocess_data(df)
    x, y = preprocessing.pd_to_numpy_X_y(x, y)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.empty(y.shape[0])
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(y)), y), 1):
        print(f"FOLD {fold}   train: {len(train_idx)}, test: {len(val_idx)}")
