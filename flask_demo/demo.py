from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get demand and stock from the form
    demand = request.form.get('demand', type=float)
    stock = request.form.get('stock', type=float)
    
    # Log request details
    app.logger.debug(f'Received request with demand: {demand}, stock: {stock}')

    input_data = pd.DataFrame([[demand, stock]], columns=['demand', 'stock'])
    
     # Predict the price
    predicted_price = model.predict(input_data)

    
    # # Predict the price
    # predicted_price = model.predict([[demand, stock]])
    
    # Log response details
    app.logger.debug(f'Predicted price: {predicted_price[0]}')

    return jsonify(predicted_price=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)







