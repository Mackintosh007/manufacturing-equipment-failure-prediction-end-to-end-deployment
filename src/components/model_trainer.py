import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for the model trainer component.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.h5")
    trained_model_history_path: str = os.path.join("artifacts", "model_history.json")
    
class ModelTrainer:
    """
    Handles the training of the machine learning model.
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr, target_encoder_path):
        """
        Initiates the model training process.
        """
        logging.info("Initiating model training process.")
        try:
            # Split the combined arrays back into features and target
            logging.info("Splitting training and testing arrays into features and target.")
            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            # Load the target encoder to determine the number of classes
            target_encoder = load_object(target_encoder_path)
            num_classes = len(target_encoder.classes_)
            
            logging.info(f"Number of features: {x_train.shape[1]}")
            logging.info(f"Number of classes: {num_classes}")
            
            # --- Define Focal Loss Function ---
            def focal_loss(gamma=2., alpha=.25):
                """
                Focal loss for multi-class classification.
                """
                def focal_loss_fixed(y_true, y_pred):
                    y_true = tf.cast(y_true, tf.int32)
                    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
                    y_true_one_hot = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=num_classes)
                    cross_entropy = -y_true_one_hot * K.log(y_pred)
                    loss_weight = y_true_one_hot * K.pow(1 - y_pred, gamma)
                    if alpha is not None:
                        alpha_factor = K.ones_like(y_true_one_hot) * alpha
                        loss_weight *= alpha_factor
                    loss = loss_weight * cross_entropy
                    return K.sum(loss, axis=-1)
                return focal_loss_fixed

            # --- Define the Model Architecture ---
            def create_lean_model(num_features, num_classes):
                """
                Creates a lean Keras Sequential model.
                """
                classifier = Sequential()
                classifier.add(layers.Dense(units=16, activation='relu', input_shape=(num_features,), kernel_regularizer=keras.regularizers.l2(0.001)))
                classifier.add(layers.Dropout(0.3))
                classifier.add(layers.Dense(units=num_classes, activation='softmax'))
                classifier.compile(optimizer='adam', loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])
                return classifier
                
            # Create the model
            model = create_lean_model(x_train.shape[1], num_classes)
            logging.info("Model created successfully.")
            
            # Define Early Stopping
            early_stopping = EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=35,
                verbose=1,
                mode="auto",
                baseline=None,
                restore_best_weights=False,
                start_from_epoch=0,
            )

            # Train the model
            logging.info("Starting model training.")
            model_history = model.fit(
                x_train,
                y_train,
                validation_split=0.33,
                batch_size=32,
                epochs=1000,
                callbacks=[early_stopping],
                verbose=1
            )
            logging.info("Model training completed.")
            
            # Save the trained model
            model.save(self.model_trainer_config.trained_model_file_path)
            logging.info(f"Trained model saved to {self.model_trainer_config.trained_model_file_path}")

            # Save the training history
            with open(self.model_trainer_config.trained_model_history_path, 'w') as f:
                json.dump(model_history.history, f)
            logging.info(f"Model training history saved to {self.model_trainer_config.trained_model_history_path}")

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # This block shows how to run the full pipeline end-to-end
    obj = DataIngestion()
    raw_data_path = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path, target_encoder_path = data_transformation.initiate_data_transformation(raw_data_path)
    
    model_trainer = ModelTrainer()
    trained_model_path = model_trainer.initiate_model_trainer(train_arr, test_arr, target_encoder_path)
    
    print(f"Model training completed. Trained model is saved at: {trained_model_path}")
