from typing import TextIO
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


def print_report_to_file(
    file: TextIO, model_name: str, y: np.ndarray, y_pred: np.ndarray
) -> None:
    file.write(f"\nClassification report for {model_name}:\n")
    report: str = str(classification_report(y, y_pred))
    file.write(report)


def print_conf_matrix_to_file(
    file: TextIO, model_name: str, y: np.ndarray, y_pred: np.ndarray
) -> None:
    file.write(f"\nConfusion matrix for {model_name}:\n")
    report: str = str(confusion_matrix(y, y_pred))
    file.write(report)


def show_images(images, cols=5, figsize=(12, 10)):
    n = len(images)
    rows = (n // cols) + 1

    plt.figure(figsize=figsize)

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray" if img.mode == "L" else None)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
