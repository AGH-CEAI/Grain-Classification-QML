from sklearn.calibration import cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from experiments import exp_run_all_class_models
import preprocessing
import models.benchmark_models as benchmark_models
import evaluation


def main():
    exp_run_all_class_models("initial_classical_results_42_mlp_new", 42)
    # model = models.get_knn_model()
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # precision_macro = make_scorer(precision_score, average="macro")
    # y_pred = cross_val_predict(model, X, y, cv=cv)


if __name__ == "__main__":
    main()
