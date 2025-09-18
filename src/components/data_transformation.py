import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    target_encoder_obj_file_path = os.path.join('artifacts', 'target_encoder.pkl')

class DataTransformation:
    """
    Handles data transformation, including outlier removal, scaling, and encoding.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, raw_data_path):
        """
        Initiates the data transformation process on the raw dataset.
        """
        try:
            df = pd.read_csv(raw_data_path)

            logging.info("Read raw data completed.")
            print(f"Shape of the initial DataFrame: {df.shape}")
            
            # Removing outliers
            logging.info("Removing outliers from Power_Consumption.")
            Q1 = df['Power_Consumption'].quantile(0.25)
            Q3 = df['Power_Consumption'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['Power_Consumption'] >= lower_bound) & (df['Power_Consumption'] <= upper_bound)]
            logging.info("Outliers removed.")
            print(f"Shape of DataFrame after outlier removal: {df.shape}")

            # Dropping columns
            logging.info("Dropping 'Equipment_ID' and separating target variable 'Failure_Cause'.")
            x = df.drop(columns=['Failure_Cause', 'Equipment_ID'], axis=1)
            y = df['Failure_Cause']
            
            print(f"Shape of features (x): {x.shape}")
            print(f"Shape of target (y): {y.shape}")

            logging.info("Performing train-test split.")

            # Perform train-test split on the pre-cleaned data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            logging.info("Train-test split completed.")
            print(f"Shape of x_train: {x_train.shape}")
            print(f"Shape of y_train: {y_train.shape}")
            logging.info("Obtaining preprocessing object and target encoder.")

            # Separating columns based on data type comes third (via the helper function) ---
            preprocessing_obj = self.get_data_transformer_object()
            target_encoder = LabelEncoder()

            logging.info("Applying preprocessing on training and testing dataframes.")

            # Apply the feature preprocessor to the data
            x_train_transformed = preprocessing_obj.fit_transform(x_train)
            x_test_transformed = preprocessing_obj.transform(x_test)
            
            # Convert sparse matrix to dense array for concatenation
            x_train_transformed = x_train_transformed.toarray()
            x_test_transformed = x_test_transformed.toarray()
            
            print(f"Shape of dense x_train_transformed: {x_train_transformed.shape}")
            print(f"Shape of dense x_test_transformed: {x_test_transformed.shape}")

            logging.info("Applying target encoding on training and testing data.")
            # Apply the target encoder to the target variable
            y_train_encoded = target_encoder.fit_transform(y_train)
            y_test_encoded = target_encoder.transform(y_test)
            
            # Reshape the 1-dimensional target arrays to a 2-dimensional column vector
            y_train_encoded = y_train_encoded.reshape(-1, 1)
            y_test_encoded = y_test_encoded.reshape(-1, 1)
            
            print(f"Shape of reshaped y_train_encoded: {y_train_encoded.shape}")
            print(f"Shape of reshaped y_test_encoded: {y_test_encoded.shape}")
            
            # Now concatenation
            train_arr = np.c_[x_train_transformed, y_train_encoded]
            test_arr = np.c_[x_test_transformed, y_test_encoded]
            
            logging.info("Saved preprocessing and target encoder objects.")

            # Save the preprocessor and target encoder objects
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            save_object(
                file_path=self.data_transformation_config.target_encoder_obj_file_path,
                obj=target_encoder
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.target_encoder_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        """
        Creates a ColumnTransformer pipeline for data transformation.
        """
        try:
            # Separate features based on their data type
            numerical_features = ['Temperature', 'Pressure', 'Vibration', 'Humidity', 'Flow_Rate', 
                                  'Power_Consumption', 'Oil_Level', 'Voltage', 'Maintenance_Cost',
                                  'Production_Volume', 'Planned_Downtime_Hours', 'Shifts_Per_Day',
                                  'Production_Days_Per_Week']
            
            categorical_features = ['Maintenance_Type', 'Installation_Date', 'Failure_Date', 'Maintenance_Date']

            # Pipeline for numerical features
            numerical_transformer = StandardScaler()

            # Pipeline for categorical features
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            
            logging.info(f"Numerical columns: {numerical_features}")
            logging.info(f"Categorical columns: {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", categorical_transformer, categorical_features),
                    ("StandardScaler", numerical_transformer, numerical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.utils import save_object
    
    obj = DataIngestion()
    raw_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path, target_encoder_path = data_transformation.initiate_data_transformation(raw_data_path)

    print(f"Shape of transformed training data: {train_arr.shape}")
    print(f"Shape of transformed testing data: {test_arr.shape}")
    print(f"Path to preprocessor object: {preprocessor_path}")
    print(f"Path to target encoder object: {target_encoder_path}")
