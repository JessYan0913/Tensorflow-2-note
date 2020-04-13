import tensorflow as tf
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

plt.imshow(x_train[0])
plt.show()

print('x_train[0]:\n', x_train[0])
print('y_train[0]:', y_train[0])
print('x_test.shape:', x_test.shape)