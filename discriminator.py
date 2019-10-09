import tensorflow as tf
from tensorflow.python.keras import layers


def make_discriminator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            use_bias=False, input_shape=(None, 64, 64, 3)))
    assert model.output_shape == (None, 64, 32, 32, 3)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                            use_bias=False))
    assert model.output_shape == (None, 128, 16, 16, 3)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same',
                            use_bias=False))
    assert model.output_shape == (None, 256, 8, 8, 3)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same',
                            use_bias=False))
    assert model.output_shape == (None, 512, 4, 4, 3)
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    return model

