import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D,GlobalAveragePooling2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal


def build_generator(SEED_SIZE, size=128):

    noise_shape = (SEED_SIZE,)
    init = RandomNormal(mean=0.0, stddev=0.02)

    model = Sequential()
  
    model.add(Dense(4 * 4 * 128, activation="relu", input_shape=noise_shape))
    model.add(Reshape((4, 4, 128)))
    model.add(BatchNormalization(momentum=0.8))
    # After this layer fake image size 4 x 4 x 128

    model.add(Conv2DTranspose(128, kernel_size=(3, 3), padding="same", strides=2, kernel_initializer=init))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    # After this layer 8 x 8 x 128

    model.add(Conv2DTranspose(128, kernel_size=(3, 3), padding="same", strides=2, kernel_initializer=init))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    # After this layer 16 x 16 x 128

    model.add(Conv2DTranspose(128, kernel_size=(3, 3), padding="same", strides=2, kernel_initializer=init))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    # After this layer 32 x 32 x 128

    if size == 64:

        model.add(Conv2DTranspose(128, kernel_size=(3, 3), padding="same", strides=2, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        # After this layer 64 x 64 x 128

    if size == 128:

        model.add(Conv2DTranspose(128, kernel_size=(3, 3), padding="same", strides=2, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        # After this layer 64 x 64 x 128

        model.add(Conv2DTranspose(128, kernel_size=(3, 3), padding="same", strides=2, kernel_initializer=init))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        # After this layer 128 x 128 x 128

    model.add(Conv2D(3, kernel_size=(3, 3), padding="same", kernel_initializer=init))
    model.add(Activation("tanh")) # because == /255 if /127.5 - 1. then use tanh
    # After this layer size x size x 3

    model.summary()
    return model

# generator = build_generator(SEED_SIZE=300, size=128)