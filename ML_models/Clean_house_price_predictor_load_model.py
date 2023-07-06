#!/usr/bin/env python
# coding: utf-8

import os
import urllib.request as request
import tarfile

import pickle
import pandas as pd
import numpy as np

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
FILE_PATH = os.path.join("datasets", "housing")
FILE_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
TGZ_NAME = "housing.tgz"


def fetch_data(file_url, file_path, file_name):
    os.makedirs(file_path, exist_ok=True)
    tgz_path = os.path.join(file_path, file_name)
    request.urlretrieve(file_url, tgz_path)
    file_tgz = tarfile.open(tgz_path)
    file_tgz.extractall(path=file_path)
    file_tgz.close()


fetch_data(FILE_URL, FILE_PATH, TGZ_NAME)

def load_data(file_path, file_name):
    csv_path = os.path.join(file_path, file_name)
    return pd.read_csv(csv_path)


CSV_NAME = "housing.csv"
df = load_data(FILE_PATH, CSV_NAME)


# make a copy of the set and create a sample set
house_df = df.copy()
sample = house_df.iloc[:5]

from app.custom_class_predictor import CombinedAttributesAdder

# load model
with open(r'C:\Users\chonl\OneDrive\Documents\GitHub\House price predictor\ML models\rndf_house_price_estimator.pkl', 'rb') as f:
    loaded_estimator = pickle.load(f)


print("Predictions:\t", loaded_estimator.predict(sample))





