import tensorflow
from matplotlib import pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as pltS


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Reshape images to add channel dimension
x_train = x_train.reshape(
    (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
)

x_test = x_test.reshape(
    (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
)


# Print shapes
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# Normalize pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0


# Create CNN model
model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    )
)

model.add(MaxPool2D(2, 2))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(10, activation='softmax'))


# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# Train model
history = model.fit(
    x_train,
    y_train,
    epochs=10
)


# Print training history
print(history)


# Evaluate model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# Select one test image
index = 54


# Display the test image
plt.imshow(
    x_test[index].reshape(28, 28),
    cmap='gray'
)

plt.title(
    "Actual digit: " + str(y_test[index])
)

plt.axis('off')
plt.show()


# Predict the selected image
y_pred = model.predict(
    x_test[index].reshape(1, 28, 28, 1)
)


# Print prediction probabilities
print("Prediction probabilities:")
print(y_pred)


# Get predicted digit
predicted_digit = y_pred.argmax()

print("Actual digit:", y_test[index])
print("Predicted digit:", predicted_digit)


# Display prediction probabilities as a bar graph
plt.bar(
    range(10),
    y_pred[0]
)

plt.xticks(range(10))
plt.xlabel("Digit")
plt.ylabel("Probability")
plt.title("Prediction Probabilities")

plt.show()