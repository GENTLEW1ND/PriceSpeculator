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

@app.route('/predict', methods=['POST','GET']) #route to show the prediction in web UI
def index():
    if request.method == 'POST':
        try:
            logging.info("Received input data from user.")
            
            #reading the inputs given by the user
            model_name = str(request.form['model_name'])
            mobile_weight = float(request.form['mobile_weight'])
            ram = float(request.form['ram'])
            front_camera = float(request.form['front_camera'])
            back_camera = float(request.form['back_camera'])
            processor = str(request.form['processor'])
            battery_capacity = float(request.form['battery_capacity'])
            screen_size = float(request.form['screen_size'])
            launched_year = int(request.form['launched_year'])
            
            logging.info(f"Parsed data: {model_name}, {mobile_weight}, {ram}, {front_camera}, {back_camera}, {processor}, {battery_capacity}, {screen_size}, {launched_year}")
            
            data = [model_name,mobile_weight,ram,front_camera,back_camera,processor,battery_capacity,screen_size,launched_year]
            data = np.array(data).reshape(1,9)
            
            logging.info(f"Reshaped data: {data}")
            
            obj = PredictionPipeline()
            predict = obj.predict(data)
            
            logging.info(f"Prediction: {predict}")
            
            return render_template('results.html', prediction = str(predict))
        
        except ValueError as ve:
            logging.error(f"ValueError: {ve}")
            return "Invalid input. Please enter valid numeric values."
        except Exception as e:
            logging.error(f"Exception occurred: {e}")
            return 'something is wrong'
    else:
        return render_template('index.html')
            
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)