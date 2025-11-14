from sklearn.calibration import cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

import experiments
import preprocessing
import models.benchmark_models as benchmark_models
import evaluation


def main():
    # exp_run_all_class_models("initial_classical_results_42_mlp_new", 42)
    experiments.exp_run_mlp(seed=42)


if __name__ == "__main__":
    main()
