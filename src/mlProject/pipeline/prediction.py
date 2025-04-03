import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"  # Define the log message format
)

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.scaler = joblib.load(Path('artifacts/model_trainer/scaler.joblib'))
        self.processor_vectorizer = joblib.load(Path("artifacts/data_transformation/processor_vectorizer.joblib"))
        self.model_vectorizer = joblib.load(Path("artifacts/data_transformation/model_vectorizer.joblib"))

    def predict(self, data):
        # Ensure the input is a Pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a Pandas DataFrame.")

        # Check for required columns
        required_columns = ['Processor', 'Model Name', 'Mobile Weight', 'RAM', 'Front Camera', 
                            'Back Camera', 'Battery Capacity', 'Screen Size', 'Launched Year']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Transform the 'Processor' column using the saved vectorizer
        logging.info("Transforming 'Processor' column...")
        processor_features = self.processor_vectorizer.transform(
            data['Processor'].fillna('').astype(str).str.lower().str.replace('-', ' ')
        )
        processor_df = pd.DataFrame(
            processor_features.toarray(), 
            columns=self.processor_vectorizer.get_feature_names_out()
        )

        # Transform the 'Model Name' column using the saved vectorizer
        logging.info("Transforming 'Model Name' column...")
        model_features = self.model_vectorizer.transform(
            data['Model Name'].fillna('').astype(str).str.lower().str.replace('-', ' ')
        )
        model_df = pd.DataFrame(
            model_features.toarray(), 
            columns=self.model_vectorizer.get_feature_names_out()
        )

        # Combine the transformed features with the rest of the data
        logging.info("Combining transformed features with the rest of the data...")
        combined_data = pd.concat([data.drop(columns=['Processor', 'Model Name']), processor_df, model_df], axis=1)
        logging.info(f"Columns in combined_data before alignment: {list(combined_data.columns)}")

        # Align the features with the training data
        expected_features = self.scaler.feature_names_in_
        logging.info(f"Expected features from scaler: {list(expected_features)}")

        # Identify missing and extra columns
        missing_columns = [col for col in expected_features if col not in combined_data.columns]
        extra_columns = [col for col in combined_data.columns if col not in expected_features]

        logging.warning(f"Missing columns in combined_data: {missing_columns}")
        logging.warning(f"Extra columns in combined_data: {extra_columns}")

        # Add missing columns
        for col in missing_columns:
            logging.warning(f"Adding missing column: {col}")
            combined_data[col] = 0  # Add missing columns with default value 0

        # Remove duplicate columns
        combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
        logging.info(f"Columns in combined_data after removing duplicates: {list(combined_data.columns)}")

        # Reorder columns to match the training data
        combined_data = combined_data[expected_features]
        logging.info(f"Columns in combined_data after alignment: {list(combined_data.columns)}")

        # Check for invalid values
        if combined_data.isnull().values.any():
            logging.error("combined_data contains NaN values. Please check the input data.")
            raise ValueError("combined_data contains NaN values.")

        if not np.isfinite(combined_data.values).all():
            logging.error("combined_data contains infinite or non-numeric values. Please check the input data.")
            raise ValueError("combined_data contains infinite or non-numeric values.")

        # Scale the combined data
        logging.info("Scaling the combined data...")
        data_scaled = self.scaler.transform(combined_data)
        data_scaled = pd.DataFrame(data_scaled, columns=combined_data.columns)

        # Convert the scaled data to a NumPy array (drop feature names)
        data_scaled_array = data_scaled.values

        # Make predictions
        logging.info("Making predictions...")
        prediction = self.model.predict(data_scaled_array)
        logging.info("Prediction completed successfully.")

        return prediction.tolist()  # Convert NumPy array to a Python list

if __name__ == "__main__":
    # Create a sample DataFrame for testing
    data = {
        'Processor': ['A17 Bionic'],
        'Model Name': ['iPhone 16 128GB'],
        'Mobile Weight': [198],
        'RAM': [8.0],
        'Front Camera': [12],
        'Back Camera': [48],
        'Battery Capacity': [3500],
        'Screen Size': [6.7],
        'Launched Year': [2024]
    }
    test_df = pd.DataFrame(data)

    # Initialize the PredictionPipeline
    pipeline = PredictionPipeline()

    # Make a prediction
    try:
        prediction = pipeline.predict(test_df)
        print(f"Prediction: {prediction}")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")