from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from typing import Tuple, List


EXCEL_DATA_LOCATION = "data/WheatGrainFeatures.xlsx"
IMAGE_DATA_LOCATION = "data/WheatGrainImages/"


def get_excel_data() -> pd.DataFrame:
    return pd.read_excel(EXCEL_DATA_LOCATION)


def get_image_data() -> Tuple[List[str], List[np.ndarray]]:
    # Get the list of all files and directories
    dir_list = os.listdir(IMAGE_DATA_LOCATION)
    imgs: List[np.ndarray] = []
    for filename in dir_list:
        with Image.open(os.path.join(IMAGE_DATA_LOCATION, filename)).convert(
            "L"
        ) as img:
            imgs.append(np.array(img))

    return dir_list, imgs
