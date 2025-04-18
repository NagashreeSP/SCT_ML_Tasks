import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 🧾 Sample dataset
data = {
    'square_feet': [1500, 1800, 2400, 3000, 3500],
    'bedrooms': [3, 4, 3, 5, 4],
    'bathrooms': [2, 2, 3, 4, 3],
    'price': [400000, 500000, 600000, 650000, 700000]
}

# 📊 Convert to DataFrame
df = pd.DataFrame(data)

# 🎯 Features and target
X = df[['square_feet', 'bedrooms', 'bathrooms']]
y = df['price']

# 🧪 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 🔮 Predictions
y_pred = model.predict(X_test)

# 📈 Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 📌 Predict new value
new_house = [[2500, 4, 3]]  # 2500 sq ft, 4 bed, 3 bath
predicted_price = model.predict(new_house)
print("Predicted price for new house:", predicted_price[0])

# Optional: visualize prediction vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
