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


def get_images_filenames(location: str = IMAGE_DATA_LOCATION) -> List[str]:
    img_list = [
        fname
        for fname in os.listdir(location)  # list all files and dirs in the location
        if os.path.isfile(os.path.join(location, fname))  # limit list only to files
    ]
    return img_list


def load_image(location: str, filename: str) -> Image.Image:
    with Image.open(os.path.join(location, filename)) as img:
        return img.copy()


def load_all_images(
    location: str = IMAGE_DATA_LOCATION,
) -> Tuple[List[str], List[Image.Image]]:
    images = []
    filenames = get_images_filenames()
    for filename in filenames:
        images.append(load_image(location, filename))
    return filenames, images
