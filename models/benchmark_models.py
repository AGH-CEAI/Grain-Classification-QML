from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# Naive Bayes (NB)
# No hyperparamenters given (maybe not needed here)
def get_nb_model():
    return GaussianNB()


# k-Nearest Neighbour (kNN)
# n_neighbors = 5 -> mentioned on the article
def get_knn_model(n_neighbors=5):
    return KNeighborsClassifier(n_neighbors=n_neighbors)


# Support Vector Machine (SVM)
# No kernel type or other hyperparameters
def get_svm_model(kernel="rbf", C=1.0):
    return SVC(kernel=kernel, C=C)  # type: ignore[arg-type]


# Logistic Regression (LR)
# No hyperparamenters given
# C, solver, penalty, dual, tol
def get_lr_model(C=1.0, max_iter=1000):
    return LogisticRegression(C=C, max_iter=max_iter)


# eXtreme Gradient Boosting (XGB)
# Hyperparameters given:
#   number of classes: 3, objective: multi:softmax, learning rate: 0.1, maximum depth: 3.
def get_xgb_model(
    objective="multi:softmax",
    num_class=3,
    learning_rate=0.1,
    max_depth=3,
):
    return XGBClassifier(
        objective=objective,
        num_class=num_class,
        learning_rate=learning_rate,
        max_depth=max_depth,
    )


# Decision Tree (DT)
# max_depth was given
def get_dt_model(max_depth: int = 24):
    return DecisionTreeClassifier(max_depth=max_depth)


# Random Forest (RF)
# No hyperparameters were given
def get_rf_model():
    return RandomForestClassifier()


# Multi-layer Perceptron (MLP)
# No info about hidden layers.
# By default: hidden_layer_sizes=(100,), learning_rate_init: Float = 0.001
# 12 input , 3 output, relu activation, adam solver, 300 max iter, batch_size=min(200, n_samples).
def get_mlp_model(
    activation="relu",
    solver="adam",
    max_iter=300,
):
    return MLPClassifier(
        activation=activation,  # type: ignore[arg-type]
        solver=solver,  # type: ignore[arg-type]
        max_iter=max_iter,
    )


model_functions = {
    "nb": get_nb_model,
    "knn": get_knn_model,
    "svm": get_svm_model,
    "lr": get_lr_model,
    "xgb": get_xgb_model,
    "dt": get_dt_model,
    "rf": get_rf_model,
    "mlp": get_mlp_model,
}
