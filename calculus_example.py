import numpy as np

# Step 1: Generate a simple dataset (synthetic data)
# We'll create a dataset with a linear relationship: y = 3x + 2 + some noise
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 data points for feature X (house size)
true_w = 3.0  # Actual weight
true_b = 2.0  # Actual bias
noise = np.random.randn(100, 1)  # Random noise to add some variation
y = true_w * X + true_b + noise   # True relationship (house prices)

# Step 2: Define our model: y_pred = w * x + b
# w and b are initialized to random values
w = np.random.randn()  # Weight initialization
b = np.random.randn()  # Bias initialization
learning_rate = 0.01  # Small step size

# Step 3: Define the number of iterations
iterations = 1000

# Step 4: Define the Mean Squared Error (MSE) loss function
def compute_loss(X, y, w, b):
    N = len(y)
    y_pred = w * X + b  # Linear model prediction
    loss = (1 / N) * np.sum((y_pred - y) ** 2)  # MSE formula
    return loss

# Step 5: Compute gradients (derivatives)
def compute_gradients(X, y, w, b):
    N = len(y)
    y_pred = w * X + b
    dw = (2 / N) * np.sum(X * (y_pred - y))  # Derivative of loss w.r.t. w
    db = (2 / N) * np.sum(y_pred - y)        # Derivative of loss w.r.t. b
    return dw, db

# Step 6: Implement Gradient Descent
loss_history = []

for i in range(iterations):
    # Calculate the current loss
    loss = compute_loss(X, y, w, b)
    loss_history.append(loss)

    # Compute gradients for w and b
    dw, db = compute_gradients(X, y, w, b)

    # Update the weights (w and b) using gradient descent
    w -= learning_rate * dw
    b -= learning_rate * db

    # Print the loss every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

# Step 7: After training, let's visualize the learned model
import matplotlib.pyplot as plt

# Plot the dataset and the learned linear model
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, w * X + b, color='red', label='Learned model')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.show()

# Final learned parameters
print(f"Final weight: {w:.4f}, Final bias: {b:.4f}")


def predict(X_new, w, b):
    return w * X_new + b  # Predict house prices using learned weight and bias

# Example: Let's predict the price for a house of size 7.5
X_new = np.array([[7.5]])  # Testing with a new house size
predicted_price = predict(X_new, w, b)
print(f"Predicted price for house size 7.5: {predicted_price[0, 0]:.2f}")

