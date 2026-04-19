import tensorflow as tf

# Initialize weights and bias
W = tf.Variable(tf.random.normal([2, 1]), dtype=tf.float32)
b = tf.Variable(tf.zeros([1]), dtype=tf.float32)

# Define model (forward pass)
def model(x):
    return tf.matmul(x, W) + b

# Example input
data = tf.constant([[1.0, 2.0]], dtype=tf.float32)

# Run the model and print output
result = model(data)
print("Output:", result.numpy())