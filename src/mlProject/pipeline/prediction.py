import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        
        self.scaler = joblib.load(Path('artifacts/model_trainer/scaler.joblib'))
        
        #Load the TF-IDF vectorizer
        self.processor_vectorizer = joblib.load(Path("artifacts/model_transformation/processor_vectorizer.joblib"))
        self.model_vectorizer = joblib.load(Path("artifacts/model_transformation/model_vectorizer.joblib"))
        
    def predict(self, data):
        
        #Transform processor column using the saved vectorizer
        processor_features = self.processor_vectorizer.transform(data['Processor'].str.lower().str.replace('-',' '))
        processor_df = pd.DataFrame(processor_features.toarray(), columns=self.processor_vectorizer.get_feature_names_out())
        
        #Transform Model Name column using the saved vectorizer
        model_features = self.model_vectorizer.transform(data['Model Name'].str.lower().str.replace('-',' '))
        model_df = pd.DataFrame(model_features.toarray(), columns=self.model_vectorizer.get_feature_names_out())
        
        # Combine the tranformed features with the rest of the data
        combined_data = pd.concat([data.drop(columns=['Processor', 'Model Name', 'Company Name']), processor_df, model_df], axis=1)
        
        data_scaled = self.scaler.transform(combined_data)
        
        prediction = self.model.predict(data_scaled)
        
        return prediction
