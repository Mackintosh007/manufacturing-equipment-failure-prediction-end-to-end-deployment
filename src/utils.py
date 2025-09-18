import os
import sys
import pickle
import joblib
import pandas as pd
from typing import Any

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_model(model_path: str):
    """
    Loads a machine learning model from a file using joblib.
    """
    if not os.path.exists(model_path):
        raise CustomException(f"Model file not found at: {model_path}", sys)

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise CustomException(f"Failed to load model from {model_path}: {e}", sys)

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Preprocesses raw input data into a pandas DataFrame.
    """
    try:
        df = pd.DataFrame([data])

        required_cols = ['Temperature', 'Pressure', 'Vibration', 'Humidity', 'Flow_Rate',
                         'Power_Consumption', 'Oil_Level', 'Voltage', 'Maintenance_Type',
                         'Maintenance_Cost', 'Production_Volume', 'Planned_Downtime_Hours',
                         'Shifts_Per_Day', 'Production_Days_Per_Week', 'Installation_Date',
                         'Failure_Date', 'Maintenance_Date']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            raise CustomException(f"Missing required input features: {missing_cols}", sys)
            
        return df
    except Exception as e:
        raise CustomException(f"Failed to preprocess input data: {e}", sys)

def postprocess_prediction(prediction: Any) -> dict:
    """
    Post-processes the model's prediction result.
    """
    try:
        result = {"prediction": prediction.tolist()}
        return result
    except Exception as e:
        raise CustomException(f"Failed to post-process prediction result: {e}", sys)
