import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

def readFile():
    dataset = pd.read_csv("landprice.csv")
    X = dataset.iloc[0: ,0].values
    Y = dataset.iloc[0: ,-1].values
    return X, Y

def split_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.4, random_state=0)
    return X_train, X_test, Y_train, Y_test

