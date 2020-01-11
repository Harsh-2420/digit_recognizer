import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np

mnist = keras.datasets.mnist

# Creating the training sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train[0])

# Displaying the images
# plt.imshow(x_train[35], cmap="gray")
# plt.show()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# # one-hot coding
# n_classes = 10
# Y_train = np_utils.to_categorical(y_train, n_classes)  # What does this do? Read docs
# Y_test = np_utils.to_categorical(y_test, n_classes)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Tested loss:", test_loss)
print("Tested Acc: ", test_acc)

model.save("digit_recognizer.model")