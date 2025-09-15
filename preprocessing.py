import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import LabelEncoder

EXCEL_DATA_LOCATION = "data/WheatGrainFeatures.xlsx"
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


def get_excel_data() -> pd.DataFrame:
    return pd.read_excel(EXCEL_DATA_LOCATION)


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


def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df = drop_columns(df)
    df = encode_nominal_data(df)
    df = add_indirect_features(df)
    # df = normalize_data(df)

    X = df[COL_FEATURES].to_numpy()
    y = df[COL_LABEL].to_numpy()
    return X, y
