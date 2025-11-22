from sklearn.calibration import cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

import experiments
import preprocessing
import models.benchmark_models as benchmark_models
import evaluation
from data.load_data import get_images_filenames, load_image, load_all_images


def main():
    # exp_run_all_class_models("initial_classical_results_42_mlp_new", 42)
    # experiments.exp_run_mlp(seed=42)

    # _, imgs = get_image_data()
    # print(imgs[10].shape)
    # print(min(img.shape[0] for img in imgs))
    # print(min(img.shape[1] for img in imgs))

    _, images = load_all_images()
    images = preprocessing.preprocess_images(images)
    evaluation.show_images(images[:20])


if __name__ == "__main__":
    main()
