import numpy as np

# Input (XOR problem)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Output
y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights randomly
np.random.seed(0)
W1 = np.random.rand(2, 2)
W2 = np.random.rand(2, 1)

# Learning rate
lr = 0.5

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training
for i in range(10000):
    
    # Forward Propagation
    hidden = sigmoid(np.dot(X, W1))
    output = sigmoid(np.dot(hidden, W2))
    
    # Error
    error = y - output
    
    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden)
    
    # Update weights
    W2 += lr * np.dot(hidden.T, d_output)
    W1 += lr * np.dot(X.T, d_hidden)

# Final Output
print("Final Output:\n", output)