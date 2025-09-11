import unittest

import preprocessing

N_RECORDS = 288
N_COLUMNS_TO_USE = 10
NUM_CATEGORIES = {0, 1, 2}


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.df = preprocessing.get_excel_data()
        self.df_drop = preprocessing.drop_columns(self.df)
        self.df_numeric = preprocessing.encode_nominal_data(self.df)

    def test_get_excel_data(self):
        self.assertEqual(self.df.shape[0], N_RECORDS)

    def test_drop_columns(self):
        for col in preprocessing.TO_DROP_COLS:
            self.assertNotIn(col, self.df_drop.columns)
        self.assertEqual(self.df_drop.columns.size, N_COLUMNS_TO_USE)

    def test_encode_nominal_data(self):
        for col in preprocessing.NOMINAL_COLS:
            for val in self.df_numeric[col].unique():
                self.assertIn(val, NUM_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
