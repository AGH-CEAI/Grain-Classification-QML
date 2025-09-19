from sklearn.calibration import cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
import preprocessing
import models
import evaluation


def main():
    df = preprocessing.get_excel_data()
    X, y = preprocessing.preprocess_data(df)
    X, y = preprocessing.pd_to_numpy_X_y(X, y)

    # model = models.get_knn_model()
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # precision_macro = make_scorer(precision_score, average="macro")
    # y_pred = cross_val_predict(model, X, y, cv=cv)

    with open("results/initial_classical_results.txt", "x") as f:
        for model_name, model_fun in models.model_functions.items():
            model = model_fun()
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            y_pred = cross_val_predict(model, X, y, cv=cv)
            evaluation.print_report_to_file(f, model_name, y, y_pred)
            evaluation.print_conf_matrix_to_file(f, model_name, y, y_pred)
            f.write(
                "\n\n******************************************************************************\n"
            )


if __name__ == "__main__":
    main()
