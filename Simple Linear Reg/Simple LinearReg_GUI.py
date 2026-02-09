import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from tkinter import *
import numpy as np

# Global model
lin_regressor = LinearRegression()

def readFile():
    dataset = pd.read_csv(r"T:\SML\Simple Linear Reg\landprice.csv")
    X = dataset[['Area']].values
    Y = dataset['Price'].values
    return X, Y

def split_data(X, Y):
    return train_test_split(X, Y, test_size=0.4, random_state=0)

def model_train():
    global lin_regressor
    X, Y = readFile()
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    lin_regressor.fit(X_train, Y_train)

def model_pred():
    area = int(entry.get())
    tran_area = np.array([[area]])     # 2D input
    pred_price = lin_regressor.predict(tran_area)[0]

    label1.config(text=f"Price: {pred_price}")
    entry.delete(0, END)

# ---------- GUI ----------
window = Tk()
window.geometry("600x700")
window.title("Land Price Predictor")

label = Label(window, text="Enter the Area of land", fg="red", font=("Courier", 15))
label.pack()

entry = Entry(window, fg='green', width=10, font=("Courier", 15))
entry.pack()

pred_button = Button(window, text="Predict", fg='red', command=model_pred, height=2, width=15)
pred_button.pack()

label1 = Label(window, fg="blue", font=("Courier", 20))
label1.pack()

# Train model once at startup
model_train()

window.mainloop()


