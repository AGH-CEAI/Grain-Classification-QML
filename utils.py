import numpy as np
import torch.utils.data as data


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
