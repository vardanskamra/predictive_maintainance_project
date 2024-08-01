import unittest
from app.preprocessing import load_data, standardize
import pandas as pd
import numpy as np
from io import StringIO
from app.logging_config import setup_logging, log_error
from sklearn.preprocessing import StandardScaler

class TestPreprocessing(unittest.TestCase):


    csv_data = """cycle,op_set_1,op_set_2,op_set_3,sensor_1,sensor_2,sensor_3,sensor_4,sensor_5,sensor_6,sensor_7,sensor_8,sensor_9,sensor_10,sensor_11,sensor_12,sensor_13,sensor_14,sensor_15,sensor_16,sensor_17,sensor_18,sensor_19,sensor_20,sensor_21,rul
1,-0.0007,-0.0004,100.0,518.67,641.82,1589.7,1400.6,14.62,21.61,554.36,2388.06,9046.19,1.3,47.47,521.66,2388.02,8138.62,8.4195,0.03,392,2388,100.0,39.06,23.419,191
2,0.0019,-0.0003,100.0,518.67,642.15,1591.82,1403.14,14.62,21.61,553.75,2388.04,9044.07,1.3,47.49,522.28,2388.07,8131.49,8.4318,0.03,392,2388,100.0,39.0,23.4236,190
3,-0.0043,0.0003,100.0,518.67,642.35,1587.99,1404.2,14.62,21.61,554.26,2388.08,9052.94,1.3,47.27,522.42,2388.03,8133.23,8.4178,0.03,390,2388,100.0,38.95,23.3442,189
4,0.0007,0.0,100.0,518.67,642.35,1582.79,1401.87,14.62,21.61,554.45,2388.11,9049.48,1.3,47.13,522.86,2388.08,8133.83,8.3682,0.03,392,2388,100.0,38.88,23.3739,188
5,-0.0019,-0.0002,100.0,518.67,642.37,1582.85,1406.22,14.62,21.61,554.0,2388.06,9055.15,1.3,47.28,522.19,2388.04,8133.8,8.4294,0.03,393,2388,100.0,38.9,23.4044,187
6,-0.0043,-0.0001,100.0,518.67,642.1,1584.47,1398.37,14.62,21.61,554.67,2388.02,9049.68,1.3,47.16,521.68,2388.03,8132.85,8.4108,0.03,391,2388,100.0,38.98,23.3669,186
7,0.001,0.0001,100.0,518.67,642.48,1592.32,1397.77,14.62,21.61,554.34,2388.02,9059.13,1.3,47.36,522.32,2388.03,8132.32,8.3974,0.03,392,2388,100.0,39.1,23.3774,185
8,-0.0034,0.0003,100.0,518.67,642.56,1582.96,1400.97,14.62,21.61,553.85,2388.0,9040.8,1.3,47.24,522.47,2388.03,8131.07,8.4076,0.03,391,2388,100.0,38.97,23.3106,184"""
    df = pd.read_csv(StringIO(csv_data))

    def test_load_data(self):
        df_test = load_data('tests/test_preprocessing.csv')
        self.assertEqual(df_test.shape[1], 26)

    def test_standardize(self):
        standardized_df, scaler_params = standardize(self.df.copy())
        self.assertEqual(standardized_df.shape, self.df.shape)
        self.assertEqual(len(scaler_params), 2)
        np.testing.assert_array_almost_equal(scaler_params[0], StandardScaler().fit(self.df.iloc[:, :-1]).mean_)
        np.testing.assert_array_almost_equal(scaler_params[1], StandardScaler().fit(self.df.iloc[:, :-1]).scale_)

        # Test standardizing data with scaler_params
        df_with_params, _ = standardize(self.df.copy(), scaler_params=scaler_params)
        pd.testing.assert_frame_equal(standardized_df, df_with_params)

    def test_standardize_invalid_scaler_params(self):
        # Test standardizing data with invalid scaler_params
        with self.assertRaises(ValueError):
            standardize(self.df.copy(), scaler_params=(1,))

if __name__ == '__main__':
    unittest.main()