import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request
from app.custom_class_predictor import CombinedAttributesAdder

flask_app = Flask(__name__)

# ML model path
model_path = "rndf_house_price_estimator.pkl"


@flask_app.route('/predict', methods=['POST'])
def model_deploy():
    try:
        longitude = request.form.get("longitude")
        latitude = request.form.get("latitude")
        housing_median_age = request.form.get("median age of houses in the area")
        total_rooms = request.form.get("total number of rooms of all houses in the area")
        total_bedrooms = request.form.get("total number of bedrooms of all houses in the area")
        population = request.form.get("population")
        households = request.form.get("number of households")
        median_income = request.form.get("median income")
        ocean_proximity = request.form.get("choose from <1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN ==> please type in capital letters")

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
            
            # load custom class that will be copied to the same folder by Dockerfile
            # load the model and pass input to the model
            predictor = pickle.load(open(model_path, 'rb'))
            X_test = pd.DataFrame([sample], columns=["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", 
                                                   "households", "median_income", "ocean_proximity"])
            prediction = predictor.predict(X_test)[0]
            return_data = {
                "error" : '0',
                "message" : 'Successfull',
                "prediction": prediction
            }
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
    return flask_app.response_class(response=json.dumps(return_data), mimetype='application/json')

@flask_app.route('/', methods=['GET'])
def root():
    return 'Welcome to the house price prediction API!'

if __name__ == "__main__":
    flask_app.run(host = '0.0.0.0', port=8080, debug=False)

