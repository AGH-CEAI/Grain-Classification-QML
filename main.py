from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import preprocessing
import models


def main():
    print("Hello, world!")
    df = preprocessing.get_excel_data()
    X, y = preprocessing.preprocess_data(df)
    print(X.dtypes[preprocessing.COL_FEATURES[0]])
    # model = models.get_knn_model()
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # precision_macro = make_scorer(precision_score, average="macro")
    # scores = cross_val_score(model, X, y, cv=cv, scoring=precision_macro)
    # print(scores.mean())


if __name__ == "__main__":
    main()
