import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

class DataTransformation:
    """
    Handles data transformation, including outlier removal, scaling, and encoding.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

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
    
    def initiate_data_transformation(self, raw_data_path):
        """
        Initiates the data transformation process on the raw dataset.
        """
        try:
            df = pd.read_csv(raw_data_path)

            logging.info("Read raw data completed.")
            logging.info("Removing outliers from Power_Consumption.")

            # Outlier removal using IQR method
            Q1 = df['Power_Consumption'].quantile(0.25)
            Q3 = df['Power_Consumption'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df['Power_Consumption'] >= lower_bound) & (df['Power_Consumption'] <= upper_bound)]
            
            logging.info("Outliers removed.")
            logging.info("Performing train-test split.")

            # Separate features and target variable
            x = df.drop(columns=['Failure_Cause', 'Equipment_ID'], axis=1)
            y = df['Failure_Cause']

            # Perform train-test split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            logging.info("Train-test split completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Apply the preprocessor to the data
            x_train_transformed = preprocessing_obj.fit_transform(x_train)
            x_test_transformed = preprocessing_obj.transform(x_test)

            # Combine transformed features and target variable
            train_arr = np.c_[x_train_transformed, np.array(y_train)]
            test_arr = np.c_[x_test_transformed, np.array(y_test)]
            
            logging.info("Saved preprocessing object.")

            # Save the preprocessor object for later use in prediction
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    
    obj = DataIngestion()
    raw_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(raw_data_path)

    print(f"Shape of transformed training data: {train_arr.shape}")
    print(f"Shape of transformed testing data: {test_arr.shape}")
    print(f"Path to preprocessor object: {preprocessor_path}")
