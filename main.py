import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate some sample data
x_train = np.random.rand(1000, 10)
y_train = np.random.randint(0, 2, size=(1000, 1))

# Define a simple neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Save the model
model.save("model.h5")

print("Model training complete and saved as 'model.h5'")
