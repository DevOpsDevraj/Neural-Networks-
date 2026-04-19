import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create dummy image dataset
X = np.random.rand(100, 64, 64, 3)
y = np.random.randint(0, 3, 100)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X, y, epochs=5, verbose=0)

# Evaluate model
loss, accuracy = model.evaluate(X, y, verbose=0)

print(f"Accuracy: {accuracy:.4f}")