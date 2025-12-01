from sklearn.calibration import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import evaluation
import logging_utils
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
            logging_utils.print_report_to_file(f, model_name, y, y_pred)
            logging_utils.print_conf_matrix_to_file(f, model_name, y, y_pred)
            f.write(
                "\n\n******************************************************************************\n"
            )


def exp_run_mlp(seed: int = 42):
    df = get_excel_data()
    X, y = preprocessing.preprocess_data(df)
    X, y = preprocessing.pd_to_numpy_X_y(X, y)
    dataset = preprocessing.get_tensor_dataset(X, y)

    args = {"input_dim": X.shape[1], "hidden_dim": 100, "output_dim": 3}
    preds, metrics = cross_val_train(
        model_cls=MLP, model_args=args, dataset=dataset, y=y, seed=seed
    )

    print(classification_report(y, preds))


# PyTorch ########################################


def exp_run_multisource_mlp(seed: int = 42):
    model_name = "MLPMultiSource"

    # Load data
    df = get_excel_data()
    ids, imgs = load_all_images()

    # Preprocessing
    tensor_features, tensor_imgs, tensor_labels = preprocessing.preprocess_all(
        df, imgs, ids
    )
    dataset = preprocessing.tensors_to_dataset(
        tensor_features, tensor_imgs, tensor_labels
    )

    logging_utils.setup_mlflow()
    with logging_utils.start_parent_run(model_name=model_name):

        hparams = {
            "input_dim_feat": tensor_features.shape[1],
            "hidden_dim_feat": 100,
            "output_dim_feat": 3,
            "in_channels": 1,
            "hidden_channels": 16,
            "hidden2_channels": 8,
            "output_img_dim": 3,
            "kernel_size": 3,
            "lr": 1e-3,
        }
        logging_utils.log_hyperparams(hparams)

        # training
        preds, metrics = cross_val_train(
            model_cls=MLPMultiSource,
            model_args=hparams,
            dataset=dataset,
            y=tensor_labels,
            seed=seed,
        )

        # logging
        logging_utils.log_aggregated_metrics(all_fold_metrics=metrics)
        logging_utils.log_classification_report(
            model_name=model_name, y_true=tensor_labels, y_pred=preds
        )
        logging_utils.log_confusion_matrix(
            model_name=model_name, y_true=tensor_labels, y_pred=preds
        )


def exp_run_quantum_multisource_mlp(seed: int = 42):
    model_name = "Quantum_MLPMultiSource"

    # Load data
    df = get_excel_data()
    ids, imgs = load_all_images()

    # Preprocessing
    tensor_features, tensor_imgs, tensor_labels = preprocessing.preprocess_all(
        df, imgs, ids
    )
    dataset = preprocessing.tensors_to_dataset(
        tensor_features, tensor_imgs, tensor_labels
    )

    logging_utils.setup_mlflow()
    with logging_utils.start_parent_run(model_name=model_name):

        hparams = {
            "input_dim_feat": tensor_features.shape[1],
            "hidden_dim_feat": 100,
            "output_dim_feat": 3,
            "in_channels": 1,
            "hidden_channels": 16,
            "hidden2_channels": 8,
            "output_img_dim": 3,
            "kernel_size": 3,
            "n_qubits": 6,
            "quantum_layers": 3,
            "lr": 1e-3,
        }
        logging_utils.log_hyperparams(hparams)

        # training
        preds, metrics = cross_val_train(
            model_cls=QuantumMLPMultiSource,
            model_args=hparams,
            dataset=dataset,
            y=tensor_labels,
            seed=seed,
        )

        # logging
        logging_utils.log_aggregated_metrics(all_fold_metrics=metrics)
        logging_utils.log_classification_report(
            model_name=model_name, y_true=tensor_labels, y_pred=preds
        )
        logging_utils.log_confusion_matrix(
            model_name=model_name, y_true=tensor_labels, y_pred=preds
        )
