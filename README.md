# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
`
1. Initialize parameters
Initialize parameters

At the beginning, the model parameters (weights and bias are initialized with small random values.
These parameters decide the slope and position of the regression line. Initialization is important because SGD gradually improves these values during training.


2.For each training data point, the model calculates the predicted output using the linear regression equation:
                           ` <img width="157" height="48" alt="image" src="https://github.com/user-attachments/assets/6fe465a6-b9d3-4b56-aa18-ad1bfcca98b5" />

Here, 
𝑥=x represents the input features, 
𝑤=w represents the weights, and 
𝑏=b is the bias.
This step gives the model’s current estimate of the output.




3.The difference between the actual output 
𝑦
y and the predicted output​is calculated using a loss function, usually Mean Squared Error (MSE):
              <img width="170" height="52" alt="image" src="https://github.com/user-attachments/assets/2a7ab143-9f74-43b3-aa56-997917852d47" />
This loss value tells how far the prediction is from the true value and helps the model understand how much it needs to adjust its parameters.


4.Update weights and bias

The weights and bias are updated using the gradient of the loss function and a learning rate 𝜂:
        <img width="247" height="85" alt="image" src="https://github.com/user-attachments/assets/2330a3c9-2d55-440c-84c9-346f671609ba" />
These updates move the parameters in the direction that reduces the error.
This process is repeated for all data points and for multiple iterations until the error is minimized.
```
## Program:
``
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: ASWINI D
RegisterNumber:25018420
from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt

# Sample data (2 features)
X = np.array([[1,2],[2,1],[3,4],[4,3],[5,5]])
y = np.array([5,6,9,10,13])

# Create model
model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')

# Train model
model.fit(X, y)

# Check learned weights
print("Weights:", model.coef_)
print("Bias:", model.intercept_)

# Predict
y_pred = model.predict(X)

# Plot Actual vs Predicted
plt.scatter(y, y_pred)
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted (SGDRegressor)")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Perfect prediction line
plt.show()
*/
```

## Output:
![Screenshot_30-1-2026_1470_127 0 0 1](https://github.com/user-attachments/assets/4c9891b5-de5e-4cd8-aa0d-bf22fdf34d93)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
