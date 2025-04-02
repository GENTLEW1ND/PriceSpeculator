from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from mlProject import logging
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__) # Initializing the flask app


@app.route('/', methods=["GET"]) # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=["GET"]) # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successfull"

@app.route('/predict', methods=['POST', 'GET']) #route to show the prediction in web UI
def index():
    if request.method == 'POST':
        try:
            logging.info("Received input data from user.")
            logging.info(f"Form data received: {request.form}")

            # Parse input data from the form
            model_name = str(request.form.get('model_name', 'unknown'))
            mobile_weight = float(request.form.get('mobile_weight', 0))
            ram = float(request.form.get('ram', 0))
            front_camera = float(request.form.get('front_camera', 0))
            back_camera = float(request.form.get('back_camera', 0))
            processor = str(request.form.get('processor', 'unknown'))
            battery_capacity = float(request.form.get('battery_capacity', 0))
            screen_size = float(request.form.get('screen_size', 0))
            launched_year = int(request.form.get('launched_year', 0))

            # Prepare the input data as a Pandas DataFrame
            data = [[processor, model_name, mobile_weight, ram, front_camera, back_camera, 
                     battery_capacity, screen_size, launched_year]]
            columns = ['Processor', 'Model Name', 'Mobile Weight', 'RAM', 'Front Camera', 
                       'Back Camera', 'Battery Capacity', 'Screen Size', 'Launched Year']
            data_df = pd.DataFrame(data, columns=columns)

            logging.info(f"Prepared DataFrame: {data_df}")

            # Make predictions
            obj = PredictionPipeline()
            predict = obj.predict(data_df)

            logging.info(f"Prediction: {predict}")

            return render_template('results.html', prediction=str(predict))

        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            return f"Invalid input: {ve}"
        except Exception as e:
            logging.error(f"Exception occurred: {e}", exc_info=True)
            return 'Something is wrong'
    else:
        return render_template('index.html')
            
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)