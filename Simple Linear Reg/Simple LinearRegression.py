import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def readFile():
    dataset = pd.read_csv("landprice.csv")
    #X = dataset.iloc[0: ,0].values
    #Y = dataset.iloc[0: ,-1].values
    X = dataset[['Area']].values    # Feature
    Y = dataset['Price'].values    # Label

    return X, Y

def split_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.4, random_state=0)
    return X_train, X_test, Y_train, Y_test

def model_train(X_train, X_test, Y_train, Y_test):
    #converting one dim array to two dim array, -1 means take care of row by yourself
    X_train1 = np.reshape(X_train, (-1,1))
    Y_train1 = np.reshape(Y_train, (-1,1))
    
    X_test1 = np.reshape(X_test, (-1,1))
    Y_test1 = np.reshape(Y_test, (-1,1))
    
    lin_regressor = LinearRegression()
    lin_regressor.fit(X_train1, Y_train1)
    
    y_pred = lin_regressor.predict(X_test1)
    print(y_pred)
    return lin_regressor

def visualize_result(X_train, X_test, Y_train, Y_test, lin_regressor):
    plt.scatter(X_train1, Y_train1, color = 'blue')
    plt.title('Area vs Price of the land')
    plt.xlabel('Area')
    plt.ylabel('price')
    
    plt.show()


def main():
    pass
