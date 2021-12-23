import pandas as pd

def get_data(nrows=1000000):
    '''returns a DataFrame with nrows '''
    df = pd.read_csv("../../data/train.csv", nrows=nrows)
    return df

def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    # A COMPLETER
    return df[(df.fare_amount.between(0, 60)) & 
            (df.passenger_count < 9 ) & 
            (df.passenger_count != 0)]