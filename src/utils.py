import os
import joblib
import pandas as pd
from exceptions import ModelLoadingError, InvalidInputError, PredictionError
from typing import Any

def load_model(model_path: str):
    
    if not os.path.exists(model_path):
        raise ModelLoadingError(f"Model file not found at: {model_path}")

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise ModelLoadingError(f"Failed to load model from {model_path}: {e}")

def preprocess_input(data: dict) -> pd.DataFrame:
    
    try:
        df = pd.DataFrame([data])

        required_cols = ['Temperature', 'Pressure', 'Vibration', 'Humidity', 'Flow_Rate', 
                         'Power_Consumption', 'Oil_Level', 'Voltage', 'Maintenance_Type', 
                         'Maintenance_Cost', 'Production_Volume', 'Planned_Downtime_Hours', 
                         'Shifts_Per_Day', 'Production_Days_Per_Week', 'Installation_Date', 
                         'Failure_Date', 'Maintenance_Date']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            raise InvalidInputError(f"Missing required input features: {missing_cols}")
            
        return df
    except Exception as e:
        raise InvalidInputError(f"Failed to preprocess input data: {e}")

def postprocess_prediction(prediction: Any) -> dict:
    
    try:
        result = {"prediction": prediction.tolist()}
        return result
    except Exception as e:
        raise PredictionError(f"Failed to post-process prediction result: {e}")
