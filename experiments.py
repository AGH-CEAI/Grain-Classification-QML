from sklearn.calibration import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import evaluation
import models.benchmark_models as benchmark_models
from models.mlp_multisource import MLPMultiSource
from models.quantum_mlp_multisource import QuantumMLPMultiSource
import preprocessing
from data.load_data import get_excel_data, load_all_images
from models.mlp import MLP
from training import cross_val_train


def exp_run_all_class_models(file_name: str, seed: int):
    df = get_excel_data()
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


def exp_run_mlp(seed: int = 42):
    df = get_excel_data()
    X, y = preprocessing.preprocess_data(df)
    X, y = preprocessing.pd_to_numpy_X_y(X, y)
    dataset = preprocessing.get_tensor_dataset(X, y)

    args = {"input_dim": X.shape[1], "hidden_dim": 100, "output_dim": 3}
    preds = cross_val_train(
        model_cls=MLP, model_args=args, dataset=dataset, y=y, seed=seed
    )

    print(classification_report(y, preds))


def exp_run_multisource_mlp(seed: int = 42):
    df = get_excel_data()
    ids, imgs = load_all_images()
    tensor_features, tensor_imgs, tensor_labels = preprocessing.preprocess_all(
        df, imgs, ids
    )
    dataset = preprocessing.tensors_to_dataset(
        tensor_features, tensor_imgs, tensor_labels
    )

    args = {"input_dim_feat": tensor_features.shape[1]}
    preds = cross_val_train(
        model_cls=MLPMultiSource,
        model_args=args,
        dataset=dataset,
        y=tensor_labels,
        seed=seed,
    )

    print(classification_report(tensor_labels, preds))


def exp_run_quantum_multisource_mlp(seed: int = 42):
    df = get_excel_data()
    ids, imgs = load_all_images()
    tensor_features, tensor_imgs, tensor_labels = preprocessing.preprocess_all(
        df, imgs, ids
    )
    dataset = preprocessing.tensors_to_dataset(
        tensor_features, tensor_imgs, tensor_labels
    )

    args = {"input_dim_feat": tensor_features.shape[1]}
    preds = cross_val_train(
        model_cls=QuantumMLPMultiSource,
        model_args=args,
        dataset=dataset,
        y=tensor_labels,
        seed=seed,
    )

    print(classification_report(tensor_labels, preds))
