import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Input data (Years of Experience)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# Output data (Salary)
y = np.array([30, 35, 45, 50, 60])

model = LinearRegression()
model.fit(X, y)

print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

y_pred = model.predict(X)

prediction = model.predict([[6]])
print("Predicted salary:", prediction[0])

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
