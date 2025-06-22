import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = yf.download("TATACONSUM.NS", start="2015-01-01", end="2023-12-31")


data['Open - Close'] = data['Open'] - data['Close']
data['High - Low'] = data['High'] - data['Low']
data = data.dropna()

plt.figure(figsize=(16,8))
plt.plot(data['Close'] , label='Closing Price')
plt.show()

# Input Features
x = data[['Open - Close', 'High - Low']]
print(x.head())


# Classification 
y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=44)

params = {'n_neighbors': list(range(2, 16))}
knn = KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)
model.fit(x_train, y_train.ravel())

print("Training Accuracy:", accuracy_score(y_train, model.predict(x_train)))
print("Testing Accuracy:", accuracy_score(y_test, model.predict(x_test)))



# Regression
y_reg = data['Close']
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x, y_reg, test_size=0.25, random_state=44)

knn_reg = KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params, cv=5)
model_reg.fit(x_train_reg, y_train_reg.values.ravel())

predictions = model_reg.predict(x_test_reg)

results = pd.DataFrame({
    "Actual_Close": np.ravel(y_test_reg),
    "Predicted_Close": np.ravel(predictions)
})

print(results.head(10))
