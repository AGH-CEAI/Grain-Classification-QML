import pandas as pd
from sklearn import preprocessing

EXCEL_DATA_LOCATION = "data/WheatGrainFeatures.xlsx"
TO_DROP_COLS = ["No.", "Id"]
NOMINAL_COLS = ["wheatvariety"]
CONTINUOUS_COLS = [
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


def get_excel_data() -> pd.DataFrame:
    return pd.read_excel(EXCEL_DATA_LOCATION)


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=TO_DROP_COLS)


def encode_nominal_data(df: pd.DataFrame) -> pd.DataFrame:
    encoder = preprocessing.LabelEncoder()
    for col in NOMINAL_COLS:
        df[col] = encoder.fit_transform(df[col])
    return df


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    for col in CONTINUOUS_COLS:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df
