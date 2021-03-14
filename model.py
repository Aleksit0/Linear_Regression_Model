from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# reading our CSV
# if CSV is not in our current dir then specify it's location
df = pd.read_csv('Salary.csv')
data = df[:-1]

# defining our training and testing features
X_train, X_test, y_train, y_test = train_test_split(data.YearsExperience, data.Salary)

# defining a Linear Regression model
model = LinearRegression()
model.fit(X_train.values.reshape(-1, 1), y_train.values)

# making a prediction(X - yrs of experience)
prediction = model.predict(X_test.values.reshape(-1, 1))

# model visualization
plt.plot(X_test, prediction, label = 'Linear Regression Model', color = 'b')
plt.scatter(X_test, y_test, label = 'Test', color = 'g', alpha = .7)
plt.legend()
plt.show()

# model accuracy
print(model.score(X_test.values.reshape(-1, 1), y_test.values))
