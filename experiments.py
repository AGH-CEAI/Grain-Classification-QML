import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from scipy.stats import ttest_rel

import logging_utils
import models.benchmark_models as benchmark_models
from models.mlp_multisource import MLPMultiSource
from models.qsvm import get_kernel_matrix_func
from models.quantum_mlp_multisource import QuantumMLPMultiSource
import preprocessing
from data.load_data import get_excel_data, load_all_images
from models.mlp import MLP
from training import cross_val_svm, cross_val_train
import evaluation


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


### PyTorch ########################################


def exp_run_mlp(seed: int = 42):
    model_name = "MLP"
    df = get_excel_data()
    X, y = preprocessing.preprocess_data(df)
    X, y = preprocessing.pd_to_numpy_X_y(X, y)
    dataset = preprocessing.get_tensor_dataset(X, y)

    logging_utils.setup_mlflow()
    with logging_utils.start_parent_run(model_name=model_name):

        hparams = {"input_dim": X.shape[1], "hidden_dim": 20, "output_dim": 3}
        logging_utils.log_hyperparams(hparams)

        preds, metrics = cross_val_train(
            model_cls=MLP, model_args=hparams, dataset=dataset, y=y, seed=seed
        )

        # logging
        logging_utils.log_aggregated_metrics(all_fold_metrics=metrics)
        logging_utils.log_classification_report(
            model_name=model_name, y_true=y, y_pred=preds
        )
        logging_utils.log_confusion_matrix(
            model_name=model_name, y_true=y, y_pred=preds
        )


def exp_run_multisource_mlp(seed: int = 42):
    model_name = "MLPMultiSource_PLUS"

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
            "hidden_dim_feat": 20,
            "output_dim_feat": 3,
            "in_channels": 1,
            "hidden_channels": 8,
            "hidden2_channels": 4,
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
            "hidden_dim_feat": 20,
            "output_dim_feat": 3,
            "in_channels": 1,
            "hidden_channels": 8,
            "hidden2_channels": 4,
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


### SVM ########################################


def exp_run_multisource_SVM(seed: int = 42):
    model_name = "SVMMultiSource"

    # Load data
    df = get_excel_data()
    ids, imgs = load_all_images()

    # Preprocessing
    tensor_features, tensor_imgs, tensor_labels = preprocessing.preprocess_all(
        df, imgs, ids
    )
    reduced_tensor_imgs = preprocessing.dim_reduction(tensor_imgs, 12)
    X = preprocessing.join_multisource(tensor_features, reduced_tensor_imgs)

    logging_utils.setup_mlflow()
    with logging_utils.start_parent_run(model_name=model_name):

        hparams = {"kernel": "rbf", "C": 1}
        logging_utils.log_hyperparams(hparams)

        # training
        preds, metrics = cross_val_svm(
            model_args=hparams, X=X, y=tensor_labels, n_splits=5, seed=seed
        )

        # logging
        logging_utils.log_aggregated_metrics(all_fold_metrics=metrics)
        logging_utils.log_classification_report(
            model_name=model_name, y_true=tensor_labels, y_pred=preds
        )
        logging_utils.log_confusion_matrix(
            model_name=model_name, y_true=tensor_labels, y_pred=preds
        )


def exp_run_quantum_multisource_SVM(seed: int = 42):
    model_name = "Quantum_SVMMultiSource"

    # Load data
    df = get_excel_data()
    ids, imgs = load_all_images()

    # Preprocessing
    tensor_features, tensor_imgs, tensor_labels = preprocessing.preprocess_all(
        df, imgs, ids
    )
    reduced_tensor_imgs = preprocessing.dim_reduction(tensor_imgs, 12)
    X = preprocessing.join_multisource(tensor_features, reduced_tensor_imgs)

    logging_utils.setup_mlflow()
    with logging_utils.start_parent_run(model_name=model_name):
        feature_map_hparams = {
            "feature_map_type": "AmplitudeEncoding",
            "n_qubits": 5,
        }
        svm_hparams = {"C": 1.0}

        logging_utils.log_hyperparams(feature_map_hparams)
        logging_utils.log_hyperparams(svm_hparams)

        qkernel = get_kernel_matrix_func(
            n_qubits=feature_map_hparams["n_qubits"], seed=seed
        )

        # training
        preds, metrics = cross_val_svm(
            model_args={"kernel": qkernel},
            X=X,
            y=tensor_labels,
            n_splits=5,
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


### TEST T-STUDENT ########################################


def two_tailed_t_test_mlp(significance_level=0.05):

    H_null = "The Quantum and Classical MLP classifiers are NOT significantly different"
    H_alt = "The Quantum and Classical MLP classifiers are significantly different"
    mlp_run_id = "1ac96b7905d34a0f9ecd11a05e0fc4e1"
    quant_mlp_run_id = "1df0d2888d5644368657e98a16321e34"
    metric_compared = "final_val_acc"

    mlp_run = evaluation.get_metrics(
        parent_run_id=mlp_run_id, metric_name=metric_compared
    )
    quant_mlp_run = evaluation.get_metrics(
        parent_run_id=quant_mlp_run_id, metric_name=metric_compared
    )

    mlp_mean_acc = np.mean(list(mlp_run.values()))
    quant_mlp_mean_acc = np.mean(list(quant_mlp_run.values()))
    diff = evaluation.get_diff_per_fold(mlp_run, quant_mlp_run)
    std = evaluation.corrected_std(diff)
    t, p = evaluation.compute_two_tailed_ttest(diff, std)

    # if p > significance_level, then we keep NULL HYPOTHESIS
    # if p > significance_level, then we keep NULL HYPOTHESIS
    if p > significance_level:
        H_true = H_null
        short_result = "Same"
    else:
        H_true = H_alt
        if mlp_mean_acc > quant_mlp_mean_acc:
            short_result = "Classical better"
        else:
            short_result = "Quantum better"

    metrics = {
        "Classical_mean_metric": mlp_mean_acc,
        "Quantum_mean_metric": quant_mlp_mean_acc,
        "T_stat": t,
        "P_value": p,
        "Significance_level": significance_level,
    }
    params = {
        "Null_Hypothesis": H_null,
        "Alternative_Hypothesis": H_alt,
        "True_Hypothesis": H_true,
        "Metric_compared": metric_compared,
        "Short_result": short_result,
    }

    return params, metrics


def two_tailed_t_test_SVM(significance_level=0.05):

    H_null = "The Quantum and Classical SVM classifiers are NOT significantly different"
    H_alt = "The Quantum and Classical SVM classifiers are significantly different"
    svm_run_id = "3766bef1473740de9e77813d5ced8d66"
    quant_svm_run_id = "c6cea6a696c74ec6ba55c67e41c9a903"
    metric_compared = "final_val_acc"

    svm_run = evaluation.get_metrics(
        parent_run_id=svm_run_id, metric_name=metric_compared
    )
    quant_svm_run = evaluation.get_metrics(
        parent_run_id=quant_svm_run_id, metric_name=metric_compared
    )

    svm_mean_acc = np.mean(list(svm_run.values()))
    quant_svm_mean_acc = np.mean(list(quant_svm_run.values()))
    diff = evaluation.get_diff_per_fold(svm_run, quant_svm_run)
    std = evaluation.corrected_std(diff)
    t, p = evaluation.compute_two_tailed_ttest(diff, std)

    # if p > significance_level, then we keep NULL HYPOTHESIS
    if p > significance_level:
        H_true = H_null
        short_result = "Same"
    else:
        H_true = H_alt
        if svm_mean_acc > quant_svm_mean_acc:
            short_result = "Classical better"
        else:
            short_result = "Quantum better"

    metrics = {
        "Classical_mean_metric": svm_mean_acc,
        "Quantum_mean_metric": quant_svm_mean_acc,
        "T_stat": t,
        "P_value": p,
        "Significance_level": significance_level,
    }
    params = {
        "Null_Hypothesis": H_null,
        "Alternative_Hypothesis": H_alt,
        "True_Hypothesis": H_true,
        "Metric_compared": metric_compared,
        "Short_result": short_result,
    }

    return params, metrics


def combined_two_tailed_ttest():
    logging_utils.setup_mlflow()
    with logging_utils.start_parent_run(model_name="Two_Tailed_Accuracy_Comparison"):
        with logging_utils.start_child_hp_run("MLP_vs_QMLP"):
            mlp_comparison_params, mlp_comparison_metrics = two_tailed_t_test_mlp()
            logging_utils.log_hyperparams(mlp_comparison_params)
            logging_utils.log_metrics(mlp_comparison_metrics)
        with logging_utils.start_child_hp_run("SVM_vs_QSVM"):
            svm_comparison_params, svm_comparison_metrics = two_tailed_t_test_mlp()
            logging_utils.log_hyperparams(svm_comparison_params)
            logging_utils.log_metrics(svm_comparison_metrics)
