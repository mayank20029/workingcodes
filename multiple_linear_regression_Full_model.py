# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/GOELMA7/Downloads/Machine Learning-A-Z-Codes-Datasets (1)/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#check for missing values
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#scaling the inp variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# Residuals
residuals = y_test - y_pred

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


#print the regression equation
equation = "y = "
for i, coef in enumerate(regressor.coef_):
    equation += f"{coef}*X{i} + "
equation += str(regressor.intercept_)
print("Regression equation:", equation)





####optional
# Assumption 1: Linearity
# Scatter plot of predicted values against residuals
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Linearity Assumption')
plt.show()

# Assumption 2: Independence (Autocorrelation)
# Durbin-Watson test
rss = np.sum(residuals**2)
tss = np.sum((y_test - np.mean(y_test))**2)
dw_test = 2 - 2 * (rss / tss)
print("Durbin-Watson statistic:", dw_test)

# Assumption 3: Homoscedasticity
# Residual plots
groups = np.random.choice([0, 1], size=100)
statistic, p_value = stats.levene(residuals, groups)
print("Levene's Test:")
print(f"Test Statistic: {statistic}")
print(f"P-value: {p_value}")

# Assumption 4: Normality
import scipy.stats as stats
# Perform Anderson-Darling test
result = stats.anderson(residuals)
# Extract test statistics and critical values
test_statistic = result.statistic
critical_values = result.critical_values
# Print the results
print(f"Anderson-Darling Test Statistic: {test_statistic}")
print("Critical Values:")
for i, cv in enumerate(critical_values):
    print(f"Level {result.significance_level[i]}: {cv}")

# Assess normality based on the test statistic
if test_statistic < critical_values[2]:
    print("The data looks approximately normally distributed.")
else:
    print("The data does not follow a normal distribution.")

plt.figure(figsize=(8, 6))
plt.hist(residuals, bins='auto', density=True)
plt.xlabel('Values')
plt.ylabel('Probability Density')
plt.title('Histogram of Data')
plt.show()


# Assumption 5: No multicollinearity
# Calculate the variance inflation factor (VIF)
print(X_train)
import statsmodels.api as sm
vif = pd.DataFrame()
X = X_train
X = sm.add_constant(X_train)
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

