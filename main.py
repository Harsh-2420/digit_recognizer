import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

# Creating the training sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Displaying the images
plt.imshow(x_train[35], cmap="gray")
plt.show()

x_train /= 255
x_test /= 255

model = keras.Sequential([
    keras.layers.Flatten((28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)