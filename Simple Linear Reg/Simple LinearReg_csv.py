import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def readFile():
    df = pd.read_csv("landprice.csv")
    X = df[['Area of Land in Thousand Sq Foot']]   # Independent variable
    Y = df['Price of Land in Million USD']         # Dependent variable
    print(df.head())
    return df

def spliData(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.4, random_state=0)
    return X_train, X_test, Y_train, Y_test

def model_train(X_train, X_test, Y_train, Y_test):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    #Make predictions    
    lin_regressor = model.predict(X_test)   
    return lin_regressor, model

def Evaluate_model(Y_test, lin_regressor):
    mse = mean_squared_error(Y_test, lin_regressor)
    r2 = r2_score(Y_test, lin_regressor)

    print("Mean Squared Error:", mse)
    print("R² Score:", r2)


def Regression_equation(model):
    print("Slope (m):", model.coef_[0])
    print("Intercept (b):", model.intercept_)
    

def visualize_result(X_train, X_test, Y_train, Y_test, lin_regressor):
    plt.scatter(X_test, Y_test)
    plt.plot(X_test, lin_regressor)
    plt.xlabel("Area of Land (Thousand Sq Foot)")
    plt.ylabel("Price of Land (Million USD)")
    plt.show()
