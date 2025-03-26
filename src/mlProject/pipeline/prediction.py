import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        
        self.scaler = joblib.load(Path('artifacts/model_trainer/scaler.joblib'))
        
    def predict(self, data):
        
        data_scaled = self.scaler.transform(data)
        
        prediction = self.model.predict(data_scaled)
        
        return prediction
