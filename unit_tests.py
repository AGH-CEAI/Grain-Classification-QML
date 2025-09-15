import unittest
import numpy as np

import preprocessing


N_RECORDS = 288
N_COLUMNS_TO_USE = 13  # 12 features + 1 label
NUM_CATEGORIES = {0, 1, 2}


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = preprocessing.get_excel_data()
        self.df_drop = preprocessing.drop_columns(self.df)
        self.df_add = preprocessing.add_indirect_features(self.df_drop)
        self.df_numeric = preprocessing.encode_nominal_data(self.df)
        self.df_normalized = preprocessing.normalize_data(self.df_add)
        self.X, self.y = preprocessing.preprocess_data(self.df)

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
        self.assertIsInstance(self.X, np.ndarray)
        self.assertIsInstance(self.y, np.ndarray)
        self.assertEqual(self.X.shape, (N_RECORDS, N_COLUMNS_TO_USE - 1))
        self.assertEqual(self.y.shape, (N_RECORDS, 1))
        self.assertTrue(np.all(np.isin(self.y, [0, 1, 2])))
        self.assertTrue(np.issubdtype(self.y.dtype, np.integer))


if __name__ == "__main__":
    unittest.main()
