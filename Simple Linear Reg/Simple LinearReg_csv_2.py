import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def readFile():
    df = pd.read_csv("landprice.csv")
    
    X = df[['Area of Land in Thousand Sq Foot']]
    Y = df['Price of Land in Million USD']
    
    return X, Y

def readFile():
    df = pd.read_csv("landprice.csv")
    
    X = df[['Area of Land in Thousand Sq Foot']]
    Y = df['Price of Land in Million USD']
    
    return X, Y

def model_train(X_train, Y_train):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def visualize_result(X_test, Y_test, model):
    Y_pred = model.predict(X_test)

    plt.scatter(X_test, Y_test)
    plt.plot(X_test, Y_pred)
    plt.xlabel("Area of Land (Thousand Sq Foot)")
    plt.ylabel("Price of Land (Million USD)")
    plt.title("Simple Linear Regression Result")
    plt.show()
    
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    print("Mean Squared Error:", mse)
    print("R² Score:", r2)

