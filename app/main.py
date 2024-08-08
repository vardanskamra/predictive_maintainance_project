import pandas as pd
import numpy as np
import pickle

from app.preprocessing import standardize, load_data
from app.logging_config import setup_logging, log_message, log_error
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

setup_logging('logs/inference.log')

try:
    loaded_model = load_model("app/models/rul_model.keras")
    with open("app/models/scaler.pkl", 'rb') as f:
        loaded_scaler = pickle.load(f)
    log_message("Loaded model and scaler successfully")
except Exception as e:
    log_error("Failed to load model/scaler: "+ str(e))

try:
    df = load_data('app/inference_data.csv') # path to data
    # inference data actual rul values: 142, 270, 79, 46, 87
    log_message("Data loaded successfully")
except Exception as e:
    log_error("Failed to load data: "+ str(e))

try:
    df=loaded_scaler.transform(df)
    log_message("Data standardized successfully")
    print(df)
except Exception as e:
    log_error("Failed to standardize data: "+str(e))

try:
    y_pred = loaded_model.predict(df)
    print(y_pred)
    log_message("Model inferencing complete")
except Exception as e:
    log_error("Model inferencing failed"+str(e))

try:
    np.savetxt('app/inference_prediction.txt', y_pred, fmt='%f') # path for predictions
    log_message("Predictions saved to inference_prediction.txt")
except Exception as e:
    log_error("Failed to save predictions: " + str(e))
