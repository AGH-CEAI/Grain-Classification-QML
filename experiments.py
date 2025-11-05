from sklearn.calibration import cross_val_predict
from sklearn.model_selection import StratifiedKFold
import evaluation
import models.benchmark_models as benchmark_models
import preprocessing


def exp_run_all_class_models(file_name: str, seed: int):
    df = preprocessing.get_excel_data()
    X, y = preprocessing.preprocess_data(df)
    X, y = preprocessing.pd_to_numpy_X_y(X, y)

    with open(f"results/{file_name}.txt", "x") as f:
        f.write(f"\nSEED: {seed}\n")
        for model_name, model_fun in benchmark_models.model_functions.items():
            model = model_fun()
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            y_pred = cross_val_predict(model, X, y, cv=cv)
            evaluation.print_report_to_file(f, model_name, y, y_pred)
            evaluation.print_conf_matrix_to_file(f, model_name, y, y_pred)
            f.write(
                "\n\n******************************************************************************\n"
            )
