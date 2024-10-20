# Example Python code to train a basic model
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score    # 0 to 1     step 1 - data collecting  , steo 2 - data train

# Sample historical data for demand and stock
data = pd.DataFrame({
    'demand': [10, 50, 30, 100, 60], 
    'stock': [100, 80, 40, 30, 60],
    'price': [10, 15, 12, 18, 16]
})

# Train the regression model
X = data[['demand', 'stock']]
y = data['price']
model = LinearRegression()
model.fit(X, y)

# Predict prices based on the training data
predicted_prices = model.predict(X)

# Calculate R² score
r2 = r2_score(y, predicted_prices)
print(f'R² score: {r2:.2f}')

# Predict new price based on new demand/stock levels
predicted_price = model.predict([[30, 230]])  # demand=40, stock=50 # type: ignore  
print(f'Predicted price: {predicted_price[0]:.2f}')

# new_data = pd.DataFrame(np.array([[40, 50]]), columns=['demand', 'stock'])
# predicted_price = model.predict(new_data)

