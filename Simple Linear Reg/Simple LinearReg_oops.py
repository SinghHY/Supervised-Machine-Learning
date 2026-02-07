import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class SimpleLinearRegression:
    
    def __init__(self, file_name):
        self.file_name = file_name
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.model = None

    def readFile(self):
        df = pd.read_csv(self.file_name)
        self.X = df[['Area of Land in Thousand Sq Foot']]
        self.Y = df['Price of Land in Million USD']

    def split_data(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=0.25, random_state=42
        )

    def model_train(self):
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.Y_train)

    def visualize_result(self):
        Y_pred = self.model.predict(self.X_test)

        plt.scatter(self.X_test, self.Y_test)
        plt.plot(self.X_test, Y_pred)
        plt.xlabel("Area of Land (Thousand Sq Foot)")
        plt.ylabel("Price of Land (Million USD)")
        plt.title("Simple Linear Regression Result")
        plt.show()

        mse = mean_squared_error(self.Y_test, Y_pred)
        r2 = r2_score(self.Y_test, Y_pred)

        print("Mean Squared Error:", mse)
        print("R² Score:", r2)

    def show_equation(self):
        print("Regression Equation:")
        print(f"Price = {self.model.coef_[0]:.3f} × Area + {self.model.intercept_:.3f}")

if __name__ == "__main__":
    slr = SimpleLinearRegression("landprice.csv")
    
    slr.readFile()
    slr.split_data()
    slr.model_train()
    slr.visualize_result()
    slr.show_equation()
