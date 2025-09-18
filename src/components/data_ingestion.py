import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        
        logging.info("Starting data ingestion process.")
        try:
            # Note: The raw data path is hardcoded here. You might want to make this configurable in the future.
            df = pd.read_csv('C:/Users/uers/Documents/model_building_excercises/manufacturing equipment maintanance/data/Dataset.csv')
            logging.info('Successfully read the dataset as a DataFrame.')

            # Create the artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the entire raw data to the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts folder.")

            logging.info("Data ingestion process completed successfully.")

            # Returns the path to the raw data for the next step in the pipeline
            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    raw_data_path = obj.initiate_data_ingestion()
    print(f"Raw data path: {raw_data_path}")
