import numpy as np

# =====================
# VECTORS
# =====================

# Vectors represent feature sets in machine learning, or input data points.
# Let's create two vectors representing two feature sets.
v1 = np.array([2, 3])
v2 = np.array([1, 4])

# 1. Vector Addition (Adding feature sets)
# In ML, this could represent combining or augmenting features from different sources.
v_add = v1 + v2
print(f"Vector Addition: {v_add}")
# Explanation: Adding two feature vectors to get the combined result (e.g., summing up features).

# 2. Dot Product (Similarity or projection of vectors)
# In ML, dot products are widely used to measure similarity between two vectors,
# e.g., between weights and input features in neural networks.
v_dot = np.dot(v1, v2)
print(f"Dot Product: {v_dot}")
# Explanation: v1 ⋅ v2 = (2 * 1) + (3 * 4) = 2 + 12 = 14.

# 3. Norm (Magnitude or size of a vector)
# Norm is used in machine learning for regularization, which helps in controlling overfitting.
v_norm = np.linalg.norm(v1)
print(f"Vector Norm: {v_norm}")
# Explanation: The norm gives the magnitude (or length) of the vector. Useful for regularization (e.g., L2 regularization).

# =====================
# MATRICES
# =====================
# Matrices are used to represent datasets (rows = samples, columns = features),
# or to represent transformations in neural networks.

# Creating two matrices representing datasets.
# Rows are data points, and columns are features.
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([[5, 6], [7, 8]])

# 4. Matrix Addition (Combine datasets or transform matrices)
# Adding two matrices can be used to combine datasets or perform basic linear transformations.
m_add = m1 + m2
print(f"Matrix Addition:\n{m_add}")
# Explanation: Adding two matrices is element-wise and helps in tasks like combining data.

# 5. Matrix Multiplication (Transformation of data)
# Matrix multiplication is essential in ML for transforming data with learned weights,
# for example in a neural network's forward pass.
m_mult = np.dot(m1, m2)
print(f"Matrix Multiplication:\n{m_mult}")
# Explanation: In neural networks, multiplying weights by input data is key to making predictions.

# 6. Matrix Transpose (Aligning dimensions for operations)
# Transposing is useful when we need to align dimensions for matrix multiplication or 
# when flipping data axes (common in PCA or covariance matrix calculations).
m_transpose = np.transpose(m1)
print(f"Matrix Transpose:\n{m_transpose}")
# Explanation: Transpose flips the matrix over its diagonal. It's important in many operations like covariance calculations.

# 7. Matrix Inverse (Solving systems of linear equations)
# Matrix inverse is used in ML, particularly in linear regression, to solve for optimal weights (parameters).
# This is useful in closed-form solutions, where we directly solve for parameters.
m_inverse = np.linalg.inv(m1)
print(f"Matrix Inverse:\n{m_inverse}")
# Explanation: The inverse is useful in linear regression to directly solve for the weights without iteration.

# =====================
# MISCELLANEOUS
# =====================

# 8. Identity Matrix (Neutral element in matrix multiplication)
# The identity matrix is like "1" for matrices. It's crucial in preserving values in transformations.
identity_matrix = np.identity(2)
print(f"Identity Matrix:\n{identity_matrix}")
# Explanation: Identity matrix is used in ML models to initialize weights or in optimization algorithms.

# =====================
# EXAMPLE: Linear Regression Closed-Form Solution
# =====================
# The closed-form solution for linear regression is w = (X^T * X)^(-1) * X^T * y

# Feature matrix (X) and output vector (y)
X = np.array([[1, 1], [1, 2], [1, 3]])  # Adding bias term
y = np.array([1, 2, 3])

# Calculating the closed-form solution
X_T = np.transpose(X)  # Transpose of the feature matrix
theta = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y)
print(f"Linear Regression Weights (theta): {theta}")
# Explanation: This calculates the optimal weights (parameters) for linear regression in one step using the formula.
