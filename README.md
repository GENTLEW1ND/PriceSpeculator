# **PriceSpeculator**

PriceSpeculator is a machine learning-based web application designed to predict smartphone prices based on their specifications. Built with Python, Flask, and Bootstrap, this tool provides an intuitive and seamless user experience for accurate predictions.

---

## **Features**
- Predict smartphone prices using inputs like **RAM**, **processor**, **camera specifications**, and more.
- **User-friendly web interface** built with Bootstrap for a smooth user experience.
- **Scalable backend** developed using Flask.
- Pre-trained machine learning model ensures fast and reliable predictions.

---

## **Installation**

### **Prerequisites**
- Python 3.8 or higher
- pip (Python package manager)

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/GENTLEW1ND/PriceSpeculator.git
   cd PriceSpeculator

2. Install dependencies:
    pip install -r requirements.txt

3. Run the application:
    python app.py

#### Project Structure
- config.yaml: Contains project configuration details.
- schema.yaml: Defines data validation schema
- params.yaml: Lists hyperparameters and other project parameters.
- src/entity/: Houses data structure definitions (entity classes)
- src/config/: Configuration management scripts
- src/components/: Contains machine learning components for data preprocessing and model training.
- src/pipeline/: Orchestrates the entire workflow
- app.py: The main Flask application for deployment.

##### Usage
Home Page
- Input smartphone details such as model name, RAM, processor, and camera specifications
- Click the "Predict" button.

Results Page
- Displays the predicted price of the smartphone.
- Option to refine input values or start a new prediction.


###### Technologies Used
- Backend: Flask
- Frontend: HTML, CSS (Bootstrap), JavaScript
- Machine Learning: Scikit-learn
- Deployment: Localhost (Deployed using Render)

###### Workflows
- Update config.yaml for configurations
- Define data schema in schema.yaml
- Modify hyperparameters and settings in params.yaml.
- Implement or refine entity classes in src/entity/
- Enhance configuration management in src/config/
- Build or update machine learning components in src/components/.
- Connect components via the pipeline in src/pipeline/
- Update and test the app in app.py.


###### License
This project is licensed under the MIT License.

###### Contact
For queries or contributions:
- Email : rajchakraborty029@gmail.com
- Github : GENTLEW1ND 