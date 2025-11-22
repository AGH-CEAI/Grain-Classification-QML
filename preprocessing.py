import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset
from PIL import Image
from torchvision import transforms


COLS_TO_DROP = ["No.", "Id"]
COL_LABEL = ["wheatvariety"]
COLS_TO_KEEP = [
    "kernelarea",
    "kernelperimeter",
    "compactness",
    "kernellength",
    "kernelwidth",
    "asymmetry",
    "groovelength",
    "germarea",
    "germlength",
]
COLS_TO_ADD = [
    "germarea_kernelarea",
    "germlength_kernellength",
    "kernelwidth_kernellength",
]
COL_FEATURES = COLS_TO_ADD + COLS_TO_KEEP


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLS_TO_DROP)


def encode_nominal_data(df: pd.DataFrame) -> pd.DataFrame:
    encoder = LabelEncoder()
    for col in COL_LABEL:
        df[col] = encoder.fit_transform(df[col])
    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in COL_FEATURES:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def add_indirect_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new["germarea_kernelarea"] = df["germarea"] / df["kernelarea"]
    df_new["germlength_kernellength"] = df["germlength"] / df["kernellength"]
    df_new["kernelwidth_kernellength"] = df["kernelwidth"] / df["kernellength"]
    return df_new


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = drop_columns(df)
    df = encode_nominal_data(df)
    df = add_indirect_features(df)
    df = normalize_data(df)

    X = df[COL_FEATURES]
    y = df[COL_LABEL]
    return X, y


def pd_to_numpy_X_y(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X_np = X.to_numpy()
    y_np = y.to_numpy().ravel()
    return X_np, y_np


def get_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)


### IMAGE PREPROCESSING ########################################


def to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")


def center_crop(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    width, height = img.size

    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    return img.crop((left, top, right, bottom))


def pad_image(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    org_width, org_height = img.size
    if target_height < org_height or target_width < org_width:
        raise ValueError(
            f"Target dimensions ({target_width}, {target_height}) can't be smaller than original dimensions ({org_width}, {org_height})"
        )
    new_image = Image.new(img.mode, (target_width, target_height))
    pad_size_left = (target_width - org_width) // 2
    pad_size_top = (target_height - org_height) // 2
    new_image.paste(img, (pad_size_left, pad_size_top))
    return new_image


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img)
    return tensor


def get_max_sizes(images: List[Image.Image]) -> Tuple[int, int]:
    w, h = [], []
    for img in images:
        w.append(img.width)
        h.append(img.height)
    return max(w), max(h)


def get_min_sizes(images: List[Image.Image]) -> Tuple[int, int]:
    w, h = [], []
    for img in images:
        w.append(img.width)
        h.append(img.height)
    return min(w), min(h)


def preprocess_images(images: List[Image.Image]):
    max_width, max_height = get_max_sizes(images)

    preprocessed_images = []
    for img in images:
        img = to_grayscale(img)
        img = pad_image(img, max_width, max_height)
        preprocessed_images.append(img)

    return preprocessed_images
