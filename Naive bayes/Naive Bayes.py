import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np

def readFile():
    dataset = pd.read_csv('T:/SML/Naive bayes/Buy_Book.csv')
    #Naive Bayes:Works with numbers, Cannot compute probabilities on strings
    #So encoding is mandatory
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
            
    X = dataset.iloc[0: ,0:4].values
    Y = dataset.iloc[0: ,-1].values
    return X, Y

def split_data(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.5, random_state=0)
    return X_train, X_test, Y_train, Y_test

def model_train(X_train, X_test, Y_train, Y_test):
    #converting one dim array to two dim array, -1 means take care of row by yourself
    X_train1 = np.reshape(X_train, (-1,1))
    Y_train1 = np.reshape(Y_train, (-1,1))
    
    X_test1 = np.reshape(X_test, (-1,1))
    Y_test1 = np.reshape(Y_test, (-1,1))
    
    classifier = GaussianNB()
    classifier.fit(X_train1, Y_train1)
    
    y_pred = classifier.predict(X_test1)
    accuracy = accuracy_score(Y_test1, y_pred)
    cm = confusion_matrix(Y_test1, y_pred)
    return classifier

def visualize_File():
    plt.scatter(X, Y, color = 'blue')
    plt.title('Age vs Buy Book')
    plt.xlabel('Age')
    plt.ylabel('Buy Book')
    
def visualize_result(X_train, X_test, Y_train, Y_test, lin_regressor):
    plt.scatter(X_test1, Y_test1 , color = 'blue')
    plt.plot(X_test1, y_pred, color = 'red')
    plt.title('Area vs Price of the land')
    plt.xlabel('Area')
    plt.ylabel('price')
    
    plt.show()


def main():
    pass
