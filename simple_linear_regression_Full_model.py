# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/GOELMA7/Downloads/Machine Learning-A-Z-Codes-Datasets (1)/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Check Missing values
missing_values = np.isnan(X)
num_missing_values = np.count_nonzero(missing_values)
print(f"Missing Values: {missing_values}")
print(f"Number of Missing Values: {num_missing_values}")

# Check Outliers
mean = np.mean(X)
std = np.std(X)
z_scores = (X - mean) / std
outliers = np.abs(z_scores) > 2
outlier_indices = np.where(outliers)[0]
print(f"Outlier Indices: {outlier_indices}")

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#model performance
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(rmse)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(r2)
averages = np.mean(y_test)
print(averages)
print(rmse*100/averages)