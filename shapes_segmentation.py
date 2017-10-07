# Segmentation demo

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from keras.models import Model
from keras.layers import Input, Conv2D
from keras.utils import plot_model
import keras.backend as K


# Generate random data

def generate_data(num):
    x, y = [], []
    for i in range(num):
        im = Image.new('RGB', (50, 50), (0, 0, 0))
        im_mask = Image.new('RGB', (50, 50), (0, 0, 0))

        draw = ImageDraw.Draw(im)
        draw_mask = ImageDraw.Draw(im_mask)

        num_shapes = 7
        edge_max = 8
        max_coord = 45

        # rectangles
        for i in range(np.random.randint(2, num_shapes)):
            edge = np.random.randint(2, edge_max)
            tl = (np.random.randint(max_coord), np.random.randint(max_coord))
            tr = (tl[0], tl[1] + edge)
            bl = (tl[0] + edge, tl[1])
            br = (tl[0] + edge, tl[1] + edge)
            draw.polygon([tl, bl, br, tr], fill="red")
            draw_mask.polygon([tl, bl, br, tr], fill="white")

        # lines
        for i in range(np.random.randint(2, num_shapes)):
            edge = np.random.randint(2, edge_max)
            a = (np.random.randint(max_coord - edge), np.random.randint(max_coord - edge))
            b = (a[0] + np.random.randint(2 * edge), a[1] + np.random.randint(2 * edge))
            draw.line((a, b), fill="red")

        # ellipses
        # for i in range(np.random.randint(2, num_shapes)):
        # 	edge = np.random.randint(2, edge_max)
        # 	tl = (np.random.randint(max_coord), np.random.randint(max_coord))
        # 	br = (tl[0] + edge, tl[1] + edge)
        # 	draw.ellipse([tl, br], fill="red")

        # pentagons
        for i in range(np.random.randint(2, num_shapes)):
            edge = np.random.randint(2, edge_max)
            tl2 = (np.random.randint(max_coord), np.random.randint(max_coord))
            tr = (tl2[0], tl2[1] + np.random.randint(edge))
            bl1 = (tl2[0] + np.random.randint(edge), tl2[1] + np.random.randint(edge))
            bl2 = (bl1[0] + np.random.randint(edge), bl1[1] + np.random.randint(edge))
            br = (bl2[0] + np.random.randint(edge), bl2[1] + np.random.randint(edge))
            draw.polygon([tl2, bl1, bl2, br, tr], fill="red")

        x.append(np.reshape(im.getdata(), (50, 50, 3)).T)
        y.append(np.reshape(im_mask.getdata(), (50, 50, 3)).T[0:1, :, :])

    x = np.array(x)
    y = np.array(y)
    return x, y


def plot_ims(model):
    x, y = generate_data(10)
    y[y == 255] = 1
    ypred = model.predict(x)

    plt.figure()
    for i in range(0, 6):
        plt.subplot(2, 6, i * 2 + 1)
        plt.imshow(x[i * 2 - 1, 0])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 6, i * 2 + 2)
        plt.imshow(ypred[i * 2 - 1, 0], cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])


def dsc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def dsc_loss(y_true, y_pred):
    return 1. - dsc(y_true, y_pred)


# Define Model

inp = Input(shape=(3, 50, 50))
l = Conv2D(8, 3, activation='relu', padding='same')(inp)
l = Conv2D(8, 3, activation='relu', padding='same')(l)
l = Conv2D(8, 3, activation='relu', padding='same')(l)
l = Conv2D(8, 3, activation='relu', padding='same')(l)
out = Conv2D(1, 1, activation='sigmoid', padding='same')(l)
model = Model(inputs=inp, outputs=out)
model.summary()
plot_model(model, to_file='shapes_model.png', show_shapes=True, show_layer_names=False)
model.compile(loss=dsc_loss, optimizer='adam', metrics=['mse', dsc])
# model.compile(loss='mse', optimizer='adam', metrics=['mse', dsc])

plot_ims(model)
plt.savefig('shapes_before_training.png')
plt.show()

eps = 5
x, y = generate_data(2000)
y[y == 255] = 1

h = model.fit(x, y, validation_split=0.1, batch_size=32, epochs=eps)

plot_ims(model)
plt.savefig('shapes_after_training.png')
plt.show()

# Plot training / validation accuracy

plt.figure()
plt.title('Shapes Segmentation')
plt.plot(range(1, eps + 1), h.history['dsc'], label='Training', color='k')
plt.plot(range(1, eps + 1), h.history['val_dsc'], label='Validation', color='r')
plt.xticks(range(1, eps + 1))
plt.xlabel('Epochs')
plt.ylabel('DICE')
plt.legend(loc='best')
plt.savefig('shapes_training.png')
plt.show()