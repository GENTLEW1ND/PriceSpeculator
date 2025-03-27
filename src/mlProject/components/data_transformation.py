from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib


class DataTransformation:
    def __init__(self, config):
        self.config = config

    def Transformation(self):
        data = pd.read_csv(self.config.unzip_data_dir,encoding='latin1')

        # Dropping unnecessary columns
        data = data.drop(columns=['Launched Price (Pakistan)', 'Launched Price (China)',
                                  'Launched Price (USA)', 'Launched Price (Dubai)'])

        # Function to clean RAM
        def remove_gb(data):
            for index, row in data.iterrows():
                if '/' in row['RAM']:
                    row_varients = row['RAM'].split('/')
                    row_values = [int(x.replace('GB', '')) for x in row_varients]
                    average_values = sum(row_values) / len(row_values)
                    data.at[index, 'RAM'] = f'{average_values}GB'
            return data

        data = remove_gb(data)
        data['RAM'] = data['RAM'].str.replace('GB', '').astype(float)

        # Convert Camera columns to string and clean values
        data['Front Camera'] = data['Front Camera'].astype(str)
        data['Back Camera'] = data['Back Camera'].astype(str)

        def process_camera(data, column_name):
            for index, row in data.iterrows():
                camera_value = row[column_name].replace('(UDC)', '')
                row_variants = camera_value.split(',') if ',' in camera_value else [camera_value]
                row_values = [int(val.replace('MP', '').replace('4K', '').strip()) for val in row_variants if val.replace('MP', '').replace('4K', '').strip().isdigit()]
                average_value = sum(row_values) / len(row_values) if row_values else 0
                data.at[index, column_name] = f'{average_value}MP'
            data[column_name] = data[column_name].str.replace('MP', '').astype(float)
            return data

        data = process_camera(data, 'Front Camera')
        data = process_camera(data, 'Back Camera')

        # Clean 'Battery Capacity'
        data['Battery Capacity'] = data['Battery Capacity'].str.replace('mAh', '').replace(',', '', regex=True).astype(float)

        # Clean 'Screen Size'
        def clean_screen_size(data, column_name):
            cleaned_values = []
            for value in data[column_name]:
                numbers = re.findall(r'\d+\.?\d*', value)
                cleaned_value = sum([float(num) for num in numbers]) / len(numbers) if numbers else float('nan')
                cleaned_values.append(cleaned_value)
            data[column_name] = cleaned_values
            return data

        data = clean_screen_size(data, 'Screen Size')

        # Clean 'Launched Price (India)'
        data['Launched Price (India)'] = data['Launched Price (India)'].str.replace('INR', '').replace(',', '', regex=True).astype(float)
        data['Mobile Weight'] = data['Mobile Weight'].str.replace('g', '').astype(float)

        # Process Processor column using TF-IDF
        processor_vectorizer = TfidfVectorizer()
        processor_features = processor_vectorizer.fit_transform(data['Processor'].str.lower().str.replace('-', ' '))
        processor_df = pd.DataFrame(processor_features.toarray(), columns=processor_vectorizer.get_feature_names_out())
        combined_features = pd.concat([data.drop(columns=['Processor']), processor_df], axis=1)
        
         # Save the processor vectorizer
        os.makedirs(self.config.root_dir, exist_ok=True)  # Ensure the directory exists
        joblib.dump(processor_vectorizer, os.path.join(self.config.root_dir, "processor_vectorizer.joblib"))

        # Process Model Name column using TF-IDF
        model_vectorizer = TfidfVectorizer()
        model_features = model_vectorizer.fit_transform(combined_features['Model Name'].str.lower().replace('-', ' '))
        model_df = pd.DataFrame(model_features.toarray(), columns=model_vectorizer.get_feature_names_out())
        df = pd.concat([combined_features.drop(columns=['Model Name', 'Company Name']), model_df], axis=1)

        joblib.dump(model_vectorizer, os.path.join(self.config.root_dir, "model_vectorizer.joblib"))
        

        return df
    

    def TrainTestSplit(self, data):
        # Split the data into training and testing sets
        train, test = train_test_split(data, test_size=0.2, random_state=42)

        
        # Save the train and test sets to CSV files
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

    
        logger.info(f"Training set shape: {train.shape}")
        logger.info(f"Test set shape: {test.shape}")

        print(train.shape)
        print(test.shape)
            
            
            
            