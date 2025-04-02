import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        
        self.scaler = joblib.load(Path('artifacts/model_trainer/scaler.joblib'))
        
        #Load the TF-IDF vectorizer
        self.processor_vectorizer = joblib.load(Path("artifacts\data_transformation\processor_vectorizer.joblib"))
        self.model_vectorizer = joblib.load(Path("artifacts\data_transformation\model_vectorizer.joblib"))
        
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
        processor_features = self.processor_vectorizer.transform(
            data['Processor'].fillna('').astype(str).str.lower().str.replace('-', ' ')
        )
        processor_df = pd.DataFrame(
            processor_features.toarray(), 
            columns=self.processor_vectorizer.get_feature_names_out()
        )

        # Transform the 'Model Name' column using the saved vectorizer
        model_features = self.model_vectorizer.transform(
            data['Model Name'].fillna('').astype(str).str.lower().str.replace('-', ' ')
        )
        model_df = pd.DataFrame(
            model_features.toarray(), 
            columns=self.model_vectorizer.get_feature_names_out()
        )

        # Combine the transformed features with the rest of the data
        combined_data = pd.concat([data.drop(columns=['Processor', 'Model Name']), processor_df, model_df], axis=1)

        # Align the features with the training data
        expected_features = self.scaler.feature_names_in_
        for col in expected_features:
            if col not in combined_data.columns:
                combined_data[col] = 0  # Add missing columns with default value 0
        combined_data = combined_data[expected_features]  # Reorder columns to match the training data

        # Scale the combined data
        data_scaled = self.scaler.transform(combined_data)

        # Make predictions
        prediction = self.model.predict(data_scaled)

        return prediction.tolist()  # Convert NumPy array to a Python list
