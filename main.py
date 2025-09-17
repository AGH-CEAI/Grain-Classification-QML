from sklearn.calibration import cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
import preprocessing
import models


def main():
    df = preprocessing.get_excel_data()
    X, y = preprocessing.preprocess_data(df)
    X, y = preprocessing.pd_to_numpy_X_y(X, y)
    model = models.get_knn_model()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # precision_macro = make_scorer(precision_score, average="macro")
    y_pred = cross_val_predict(model, X, y, cv=cv)

    # Per-class precision/recall/F1
    print(classification_report(y, y_pred))

    # Confusion matrix
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    main()
