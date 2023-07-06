import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
# load custom class that will be copied to the same folder by Dockerfile
from custom_class_predictor import CombinedAttributesAdder



flask_app = Flask(__name__)

# ML model path
model_path = "rndf_house_price_estimator.pkl"
# load the model and pass input to the model
predictor = pickle.load(open(model_path, 'rb'))

@flask_app.route('/')
def home():
    return render_template("home.html")

@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        longitude = request.form.get("longitude")
        latitude = request.form.get("latitude")
        housing_median_age = request.form.get("housing_median_age")
        total_rooms = request.form.get("total_rooms")
        total_bedrooms = request.form.get("total_bedrooms")
        population = request.form.get("population")
        households = request.form.get("households")
        median_income = request.form.get("median_income")
        ocean_proximity = request.form.get("ocean_proximity")

        fields = [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population,
                  households, median_income, ocean_proximity]
        if not None in fields:
            # convert input to float
            longitude = float(longitude)
            latitude = float(latitude)
            housing_median_age = float(housing_median_age)
            total_rooms = float(total_rooms)
            total_bedrooms = float(total_bedrooms)
            population = float(population)
            households = float(households)
            median_income = float(median_income)
            sample = [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, 
                      households, median_income, ocean_proximity]
            
            X_test = pd.DataFrame([sample], columns=["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", 
                                                   "households", "median_income", "ocean_proximity"])
            return_data = predictor.predict(X_test)[0]
            print(return_data)
        else:
            return_data = {
                "error" : '1',
                "message" : 'Invalid inputs',
                "fields": fields
            }
    except Exception as e:
        return_data = {
            'error': '2',
            'message': str(e),
        }
    return render_template('home.html', prediction_text="House median value is ${}".format(return_data))

@flask_app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    sample = request.get_json(force=True)
    X_test = pd.DataFrame([sample], columns=["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", 
                                              "households", "median_income", "ocean_proximity"])
    prediction = predictor.predict(X_test)[0]

    return_data = prediction
    return jsonify(return_data)

if __name__ == "__main__":
    flask_app.run(host = '0.0.0.0', port=8080, debug=False)

