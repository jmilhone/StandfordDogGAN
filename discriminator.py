import tensorflow as tf
from tensorflow.python.keras import layers

weights_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02, mean=0.0)

def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            use_bias=False, input_shape=(64, 64, 3), kernel_initializer=weights_initializer))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                            use_bias=False, kernel_initializer=weights_initializer))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same',
                            use_bias=False, kernel_initializer=weights_initializer))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same',
                            use_bias=False, kernel_initializer=weights_initializer))
    assert model.output_shape == (None, 4, 4, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))#, activation='sigmoid'))
    return model


if __name__ == "__main__":
    discriminator = make_discriminator_model()
