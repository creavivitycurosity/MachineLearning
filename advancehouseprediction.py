import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('data.csv')

# Convert date to useful features
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Drop columns that are not needed
df = df.drop(['date', 'street', 'country'], axis=1)

# Label encode categorical variables like 'city' and 'statezip'
label_encoder = LabelEncoder()
df['city'] = label_encoder.fit_transform(df['city'])
df['statezip'] = label_encoder.fit_transform(df['statezip'])

# Define features and target variable
X = df.drop('price', axis=1)  # Features
y = df['price']               # Target variable (price)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Function to predict house price based on user input
def predict_price(input_data):
    # Predict the price
    predicted_price = model.predict([input_data])
    return predicted_price[0]

# Example input (you can change this to test)
# Updated input (now includes 16 features)
example_input = [3.0, 2.0, 2000, 10000, 1.0, 0, 0, 3, 1500, 500, 1995, 0, 98133, 1, 2014, 5]  # Adjust these values
predicted_price = predict_price(example_input)
print("Predicted House Price for the input:", predicted_price)
