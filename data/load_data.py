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


def get_images_paths() -> List[str]:
    dir_list = [
        os.path.join(IMAGE_DATA_LOCATION, fname)  # get the full path to file
        for fname in os.listdir(
            IMAGE_DATA_LOCATION
        )  # list all files and dirs in the location
        if os.path.isfile(
            os.path.join(IMAGE_DATA_LOCATION, fname)
        )  # limit list only to files
    ]

    return dir_list


def load_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.copy()
