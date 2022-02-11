import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal


def build_discriminator(image_shape, size=128):
    init = RandomNormal(mean=0.0, stddev=0.02)

    model = Sequential()

    if size == 128:
        # 128, 128, 3
        model.add(Conv2D(32, (3, 3), strides=2,
                padding="same", input_shape=image_shape))
        model.add(LeakyReLU(alpha=0.2))
        # After this 64 x 64 x 32

        model.add(Dropout(rate=0.25))
        model.add(Conv2D(128, (3, 3), strides=2,
                padding="same", kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # After this 32 x 32 x 128

        model.add(Dropout(rate=0.25))
        model.add(Conv2D(128, (3, 3), strides=2,
                padding="same", kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # After this 16 x 16 x 128

        model.add(Dropout(rate=0.25))
        model.add(Conv2D(128, (3, 3), strides=2,
                padding="same", kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # After this 8 x 8 x 128

        model.add(Dropout(rate=0.25))
        model.add(Conv2D(128, (3, 3), strides=2,
                padding="same", kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # After this 4 x 4 x 128

        model.add(Dropout(rate=0.25))
        model.add(Conv2D(128, (3, 3), strides=1,
                padding="same", kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # After this 4 x 4 x 128

    if size == 64:
        # 64, 64, 3
        model.add(Conv2D(32, (3, 3), strides=2,
                padding="same", input_shape=image_shape))
        model.add(LeakyReLU(alpha=0.2))
        # After this 32 x 32 x 32

        model.add(Dropout(rate=0.25))
        model.add(Conv2D(128, (3, 3), strides=2,
                padding="same", kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # After this 16 x 16 x 128

        model.add(Dropout(rate=0.25))
        model.add(Conv2D(128, (3, 3), strides=2,
                padding="same", kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # After this 8 x 8 x 128

        model.add(Dropout(rate=0.25))
        model.add(Conv2D(128, (3, 3), strides=2,
                padding="same", kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # After this 4 x 4 x 128
    
    if size == 32:
        # 32, 32, 3
        model.add(Conv2D(32, (3, 3), strides=2,
                padding="same", input_shape=image_shape))
        model.add(LeakyReLU(alpha=0.2))
        # After this 16 x 16 x 32

        model.add(Dropout(rate=0.25))
        model.add(Conv2D(128, (3, 3), strides=2,
                padding="same", kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # After this 8 x 8 x 128

        model.add(Dropout(rate=0.25))
        model.add(Conv2D(128, (3, 3), strides=2,
                padding="same", kernel_initializer=init))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # After this 4 x 4 x 128

    model.add(Flatten())  # 4*4*128

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.summary()

    return model


# discriminator = build_discriminator(image_shape=(32, 32, 3), size=32)