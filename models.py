from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# Naive Bayes (NB)
def get_nb_model():
    return GaussianNB()


# k-Nearest Neighbour (kNN)
def get_knn_model(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)


# Support Vector Machine (SVM)
def get_svm_model(kernel=["rbf"], C=1.0):
    return SVC(kernel=kernel, C=C)


# Logistic Regression (LR)
def get_lr_model(C=1.0, max_iter=1000):
    return LogisticRegression(C=C, max_iter=max_iter)


# eXtreme Gradient Boosting (XGB)
def get_xgb_model(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss"):
    return XGBClassifier(
        n_estimators=n_estimators,
        use_label_encoder=use_label_encoder,
        eval_metric=eval_metric,
    )


# Decision Tree (DT)
def get_dt_model(max_depth=None):
    return DecisionTreeClassifier(max_depth=max_depth)


# Random Forest (RF)
def get_rf_model(n_estimators=100, max_depth=None):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)


# Multi-layer Perceptron (MLP)
def get_mlp_model(hidden_layer_sizes=(100,), max_iter=200):
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
