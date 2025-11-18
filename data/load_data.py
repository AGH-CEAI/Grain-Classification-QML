import pandas as pd

EXCEL_DATA_LOCATION = "data/WheatGrainFeatures.xlsx"


def get_excel_data() -> pd.DataFrame:
    return pd.read_excel(EXCEL_DATA_LOCATION)
