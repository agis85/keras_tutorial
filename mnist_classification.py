# MNIST Demo

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import plot_model

# Load MNIST

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train, axis=1)
y_train = np.expand_dims(y_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)
y_test = np.expand_dims(y_test, axis=1)

x_train = x_train[0:20000]
y_train = y_train[0:20000]

enc = OneHotEncoder()
enc.fit(np.array(y_train))
y_train = enc.transform(y_train).toarray()
enc.fit(y_test)
y_test = enc.transform(y_test).toarray()


def plot_ims(model):
    y_pred = model.predict(x_test[0:12])
    plt.figure()
    for i in range(1, 13):
        plt.subplot(3, 4, i)
        plt.imshow(x_test[i - 1, 0], cmap=plt.cm.gray)
        plt.title(np.argmax(y_pred[i - 1]))
        plt.xticks([])
        plt.yticks([])


# Define Model

inp = Input(shape=(1, 28, 28))
l = Conv2D(8, 3, activation='relu')(inp)
l = Conv2D(8, 3, activation='relu')(l)
l = Flatten()(l)
out = Dense(10, activation='softmax')(l)
model = Model(inputs=inp, outputs=out)
model.summary()
plot_model(model, to_file='mnist_model.png', show_shapes=True, show_layer_names=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

plot_ims(model)
plt.savefig('mnist_before_training.png')
plt.show()

eps = 5
h = model.fit(x_train, y_train, validation_split=0.1, batch_size=32, epochs=eps)

plot_ims(model)
plt.savefig('mnist_after_training.png')
plt.show()

# Plot training / validation accuracy

plt.figure()
plt.title('MNIST Classification')
plt.plot(range(1, eps + 1), h.history['acc'], label='Training', color='k')
plt.plot(range(1, eps + 1), h.history['val_acc'], label='Validation', color='r')
plt.xticks(range(1, eps + 1))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.savefig('mnist_training.png')
plt.show()