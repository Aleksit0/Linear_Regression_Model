from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('Salary.csv')
data = df[:-1]

X_train, X_test, y_train, y_test = train_test_split(data.YearsExperience, data.Salary)

regresor = LinearRegression()
regresor.fit(X_train.values.reshape(-1, 1), y_train.values)

prediction = regresor.predict(X_test.values.reshape(-1, 1))

plt.plot(X_test, prediction, label = 'Linear Regression Model', color = 'b')
plt.scatter(X_test, y_test, label = 'Test', color = 'g', alpha = .7)
plt.legend()
plt.show()

print(regresor.score(X_test.values.reshape(-1, 1), y_test.values))
