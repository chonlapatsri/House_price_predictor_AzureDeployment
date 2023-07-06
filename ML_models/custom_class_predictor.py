from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator
import numpy as np

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_idx, self.bedrooms_idx, self.population_idx, self.households_idx = 3, 4, 5, 6
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, self.rooms_idx]/X[:, self.households_idx]
        population_per_household = X[:, self.population_idx]/X[:, self.households_idx]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_idx]/X[:, self.rooms_idx]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]