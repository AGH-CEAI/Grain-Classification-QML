from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import t
import mlflow
import pandas as pd

### IMAGES ########################################


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


### STATISTICAL COMPARISON ########################################


def get_metrics(parent_run_id: str, metric_name: str) -> dict:
    # mlflow.get_run(run_id=parent_run_id).data.to_dictionary()
    df = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'"
    )
    per_fold_metrics = {}
    for _, row in df.iterrows():  # type: ignore
        per_fold_metrics[row["tags.mlflow.runName"]] = row[f"metrics.{metric_name}"]

    # Error handling: ensure we found metrics for the given parent run and metric name
    if not per_fold_metrics:
        raise ValueError(
            f"No metrics found for parent_run_id='{parent_run_id}' and metric_name='{metric_name}'"
        )
    return per_fold_metrics


def get_diff_per_fold(metrics_1: dict, metrics_2: dict):
    diff = []
    for key in metrics_1.keys():
        diff.append(metrics_1[key] - metrics_2[key])
    return diff


def corrected_std(differences, n_train=230, n_test=58):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """

    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_two_tailed_ttest(differences, std):
    """
    Returns
    -------
    t_stat, p_value
    """
    mean = np.mean(differences)
    df = len(differences) - 1
    t_stat = mean / std
    p_val = 2 * t.sf(np.abs(t_stat), df)  # two-tailed t-test
    return t_stat, p_val
