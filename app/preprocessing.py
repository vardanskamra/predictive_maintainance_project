import pandas as pd
import pickle
from app.logging_config import log_error
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the dataset from a given file path.

    Parameters:
    - file_path (str): The path to the data file.

    Returns:
    - pd.DataFrame: The loaded dataset.
    """

    df = pd.read_csv(file_path)
    return df
    

def standardize(df, scaler_params=None, save_path=None):
    """
    Standardize the dataset (Z-score normalization).

    Parameters:
    - df (pd.DataFrame): The dataset to standardize using standard scaler.
    - scaler_params(tuple(scaler.mean_, scaler.scale_)): Tuple with scaler parameter arrays (optional)

    Returns:
    - pd.DataFrame: The standardized dataset.
    - scaler_params: Tuple with scaler parameters (mean, scale)
    """

    # Extract columns to standardize and the last column
    columns_to_standardize = df.columns[:-1]
        
    scaler = StandardScaler()

    if scaler_params:
        # Check if scaler_params is a tuple with two elements
        if len(scaler_params) != 2:
            raise ValueError("scaler_params must be a tuple with two elements: (mean, scale)")
        scaler.mean_, scaler.scale_ = scaler_params
        df[columns_to_standardize] = scaler.transform(df[columns_to_standardize])
    else:
        # Fit and transform the data
        df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
        scaler_params = (scaler.mean_, scaler.scale_)

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)

    return df, scaler_params

    



    


