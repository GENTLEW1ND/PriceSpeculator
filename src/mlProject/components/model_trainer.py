import pandas as pd
import os
from mlProject import logging
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.preprocessing import StandardScaler
from mlProject.entity.config_entity import ModelTrainerConfig 

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column],axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_x)
        test_scaled = scaler.transform(test_x)
        
        os.makedirs(self.config.root_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(self.config.root_dir, "scaler.joblib"))
        
         # Save the transformed test data
        transformed_test_data = pd.DataFrame(test_scaled, columns=train_x.columns)
        transformed_test_data[self.config.target_column] = test_y.reset_index(drop=True)
        transformed_test_data.to_csv(os.path.join(self.config.root_dir, "transformed_test_data.csv"), index=False)
        
        lr = RandomForestRegressor(n_estimators=self.config.n_estimators, max_depth=self.config.max_depth, min_samples_leaf=self.config.min_samples_leaf, min_samples_split=self.config.min_samples_split)
        lr.fit(train_scaled,train_y)
        
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))