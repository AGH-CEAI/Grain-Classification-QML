import experiments


def main():
    # _, imgs = get_image_data()
    # print(imgs[10].shape)
    # print(min(img.shape[0] for img in imgs))
    # print(min(img.shape[1] for img in imgs))

    # exp_run_all_class_models("initial_classical_results_42_mlp_new", 42)
    # experiments.exp_run_mlp(seed=42)

    experiments.exp_run_multisource_mlp(seed=42)
    # experiments.exp_run_quantum_multisource_mlp(seed=42)
    # experiments.exp_run_multisource_SVM(seed=42)
    # experiments.exp_run_quantum_multisource_SVM(seed=42)


if __name__ == "__main__":
    main()
