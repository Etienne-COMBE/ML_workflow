from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import utils

class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):

        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        # A COMPPLETER
    
    def fit(self, X, y=None):
        return self
        # A COMPLETER 

    def transform(self, X, y=None):
        return pd.DataFrame(utils.haversine_vectorized(X), columns=["distance"])

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self, datetime="pickup_datetime", time_zone ="America/New_York"):
        self.datetime = datetime    
        self.time_zone = time_zone
       # A COMPLETER

    def fit(self, X, y=None):
        return self
        # A COMPLETER

    def transform(self, X, y=None):
        X[self.datetime] = pd.to_datetime(X[self.datetime], format="%Y-%m-%d %H:%M:%S %Z").dt.tz_convert(self.time_zone)
        X_ = pd.DataFrame()
        X_["dow"] = X[self.datetime].dt.dayofweek
        X_["hour"] = X[self.datetime].dt.hour
        X_["month"] = X[self.datetime].dt.month
        X_["year"] = X[self.datetime].dt.year

        # A COMPLETER
        return X_