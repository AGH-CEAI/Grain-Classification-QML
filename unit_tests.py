import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

import preprocessing
import evaluation
from data.load_data import (
    get_excel_data,
    get_images_filenames,
    load_all_images,
    load_image,
)
from PIL import Image

N_RECORDS = 288
N_COLUMNS_TO_USE = 13  # 12 features + 1 label
NUM_CATEGORIES = {0, 1, 2}


class TestExcelDataLoadAndPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = get_excel_data()
        self.df_drop = preprocessing.drop_columns(self.df)
        self.df_add = preprocessing.add_indirect_features(self.df_drop)
        self.df_numeric = preprocessing.encode_nominal_data(self.df)
        self.df_normalized = preprocessing.normalize_data(self.df_add)
        self.X, self.y = preprocessing.preprocess_data(self.df)
        self.X_np, self.y_np = preprocessing.pd_to_numpy_X_y(self.X, self.y)
        self.tensor_ds = preprocessing.get_tensor_dataset(self.X_np, self.y_np)
        self.tensor_ds_2 = preprocessing.get_tensor_dataset(self.X_np, self.y_np)

    def test_get_excel_data(self):
        self.assertEqual(self.df.shape[0], N_RECORDS)

    def test_drop_columns(self):
        for col in preprocessing.COLS_TO_DROP:
            self.assertNotIn(col, self.df_drop.columns)
        self.assertEqual(
            self.df_drop.columns.size, N_COLUMNS_TO_USE - len(preprocessing.COLS_TO_ADD)
        )

    def test_encode_nominal_data(self):
        for col in preprocessing.COL_LABEL:
            for val in self.df_numeric[col].unique():
                self.assertIn(val, NUM_CATEGORIES)

    def test_normalize_data(self):
        for col in preprocessing.COL_FEATURES:
            self.assertGreaterEqual(self.df_normalized[col].min(), 0)
            self.assertLessEqual(self.df_normalized[col].max(), 1)

    def test_add_indirect_features(self):
        self.assertEqual(self.df_add.columns.size, N_COLUMNS_TO_USE)
        for col in preprocessing.COL_FEATURES:
            self.assertIn(col, self.df_add.columns)
        self.assertIn(preprocessing.COL_LABEL[0], self.df_add.columns)

    def test_preprocess_data(self):
        self.assertIsInstance(self.X, pd.DataFrame)
        self.assertIsInstance(self.y, pd.DataFrame)
        self.assertEqual(self.X.shape, (N_RECORDS, N_COLUMNS_TO_USE - 1))
        self.assertEqual(self.y.shape, (N_RECORDS, 1))
        # Check for empty cells
        self.assertTrue((self.X.isnull().values == False).all())
        self.assertTrue((self.y.isnull().values == False).all())
        # Check if correct labaling was used
        self.assertTrue(self.y[preprocessing.COL_LABEL[0]].isin([0, 1, 2]).all())
        # Check if all features and labels are of correct type
        self.assertEqual(self.X.dtypes.unique().size, 1)
        self.assertEqual(self.y.dtypes.unique().size, 1)
        self.assertIsInstance(self.X[preprocessing.COL_FEATURES[0]][0], np.float64)
        self.assertIsInstance(self.y[preprocessing.COL_LABEL[0]][0], np.integer)

    def test_pd_to_numpy_X_Y(self):
        self.assertIsInstance(self.X_np, np.ndarray)
        self.assertIsInstance(self.y_np, np.ndarray)

    def test_get_tensor_dataset(self):
        self.assertIsInstance(self.tensor_ds, TensorDataset)
        self.assertEqual(self.tensor_ds.tensors[0].shape[0], self.X_np.shape[0])
        self.assertEqual(self.tensor_ds.tensors[0].shape[1], self.X_np.shape[1])
        self.assertEqual(self.tensor_ds.tensors[1].shape[0], self.y_np.shape[0])

        self.assertIsInstance(self.tensor_ds_2, TensorDataset)
        self.assertEqual(self.tensor_ds_2.tensors[0].shape[0], self.X.shape[0])
        self.assertEqual(self.tensor_ds_2.tensors[0].shape[1], self.X.shape[1])
        self.assertEqual(self.tensor_ds_2.tensors[1].shape[0], self.y.shape[0])


class TestTraining(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()


class TestImageLoadAndPreprocessing(unittest.TestCase):

    def setUp(self) -> None:
        self.imgs_names, self.images = load_all_images()
        self.img_example = self.images[0]

    def test_load_images(self):
        self.assertEqual(len(self.imgs_names), N_RECORDS)
        self.assertEqual(type(self.img_example), Image.Image)

    def test_to_grayscale(self):
        gray_img = preprocessing.to_grayscale(self.img_example)
        self.assertEqual(gray_img.mode, "L")
        self.assertEqual(gray_img.size, self.img_example.size)

    def test_center_crop(self):
        cropped = preprocessing.center_crop(self.img_example, 50, 50)
        self.assertEqual(cropped.size, (50, 50))

        # Test that crop is centered
        left, top = (self.img_example.width - 50) // 2, (
            self.img_example.height - 50
        ) // 2
        right, bottom = left + 50, top + 50

        cropped_pixels = cropped.load()
        original_pixels = self.img_example.load()

        for i in range(50):
            for j in range(50):
                self.assertEqual(
                    cropped_pixels[i, j], original_pixels[i + left, j + top]  # type: ignore
                )

    def test_pad_image(self):
        padded_image = preprocessing.pad_image(self.img_example, 850, 850)
        self.assertEqual((850, 850), padded_image.size)

    def test_image_to_tensor(self):
        tensor = preprocessing.image_to_tensor(self.img_example)
        # Check tensor shape: (C, H, W)
        expected_shape = (
            len(self.img_example.getbands()),
            self.img_example.height,
            self.img_example.width,
        )
        self.assertEqual(tensor.shape, expected_shape)
        # Check dtype
        self.assertTrue(torch.is_tensor(tensor))
        self.assertTrue(tensor.dtype == torch.float32)
        # Check values are in [0, 1]
        self.assertTrue(tensor.max() <= 1.0 and tensor.min() >= 0.0)


class TestPreprocessingUtilities(unittest.TestCase):

    def test_sort_dataframe(self):
        # Mock data
        df = pd.DataFrame({"id": ["b", "a", "c"], "value": [2, 1, 3]})
        ids = ["a", "b", "c"]

        sorted_df = preprocessing.sort_dataframe(df, ids)

        # Check order
        self.assertEqual(sorted_df["id"].tolist(), ids)
        # Check values are aligned
        self.assertEqual(sorted_df["value"].tolist(), [1, 2, 3])


@patch("data.preprocessing.preprocess_images")
@patch("data.preprocessing.preprocess_data")
def test_preprocess_all(self, mock_preprocess_data, mock_preprocess_images):
    # -------- Input data --------
    df = pd.DataFrame(
        {
            "id": ["b", "a", "c"],
            "feat1": [10, 20, 30],
            "feat2": [1.0, 2.0, 3.0],
            "label": [0, 1, 0],
        }
    )
    imgs = [
        Image.new("L", (1, 1)),
        Image.new("L", (1, 1)),
        Image.new("L", (1, 1)),
    ]  # placeholders
    ids = ["a", "b", "c"]  # desired order

    # -------- Mock behavior --------
    mock_preprocess_images.return_value = torch.randn(3, 1, 100, 100)

    # preprocess_data returns Xdf and ydf
    mock_x = pd.DataFrame({"feat1": [20, 10, 30], "feat2": [2, 1, 3]})
    mock_y = pd.Series([1, 0, 0])
    mock_preprocess_data.return_value = (mock_x, mock_y)

    # -------- Call function --------
    tensor_x, imgs_tensor, tensor_y = preprocessing.preprocess_all(df, imgs, ids)

    # -------- Assertions --------
    # images called correctly
    mock_preprocess_images.assert_called_once_with(imgs)

    # df was sorted BEFORE preprocess_data
    expected_sorted = df.set_index("id").loc[ids].reset_index()
    mock_preprocess_data.assert_called_once()
    pd.testing.assert_frame_equal(mock_preprocess_data.call_args[0][0], expected_sorted)

    # returned objects are tensors
    self.assertIsInstance(tensor_x, torch.Tensor)
    self.assertIsInstance(tensor_y, torch.Tensor)
    self.assertIsInstance(imgs_tensor, torch.Tensor)

    # sizes match
    self.assertEqual(len(tensor_x), len(tensor_y) == 3)
    self.assertEqual(len(imgs_tensor), 3)

    def test_tensors_to_dataset(self):
        features = torch.randn(5, 3)
        imgs = torch.randn(5, 1, 10, 10)
        labels = torch.randint(0, 2, (5,))

        dataset = preprocessing.tensors_to_dataset(features, imgs, labels)

        # Check dataset length
        self.assertEqual(len(dataset), 5)

        # Check alignment
        f, i, l = dataset[2]
        self.assertTrue(torch.equal(f, features[2]))
        self.assertTrue(torch.equal(i, imgs[2]))
        self.assertTrue(torch.equal(l, labels[2]))


if __name__ == "__main__":
    unittest.main(buffer=False)
