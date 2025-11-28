from typing import TextIO
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import mlflow
from statistics import mean, stdev


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


def start_parent_run(model_name: str, dataset_name: str):
    parent = mlflow.start_run(run_name=model_name)
    mlflow.set_tag("model", model_name)
    mlflow.set_tag("dataset", dataset_name)
    return parent


def log_hyperparams(params):
    mlflow.log_params(params)


def start_child_hp_run(hp_id: str):
    return mlflow.start_run(run_name=f"hp_{hp_id}", nested=True)


def log_fold_metrics(fold_idx, metrics: dict):
    mlflow.log_param("fold", fold_idx)
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_aggregated_metrics(all_fold_metrics: dict):
    for metric_name, values in all_fold_metrics.items():
        mlflow.log_metric(f"{metric_name}_mean", mean(values))
        mlflow.log_metric(f"{metric_name}_std", stdev(values))
