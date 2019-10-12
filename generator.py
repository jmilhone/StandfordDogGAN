import tensorflow as tf
from tensorflow.python.keras import layers

weight_initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02, mean=0.0)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4 * 4 * 1024, use_bias=False, input_shape=(100, ), kernel_initializer=weight_initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 1024)))
    assert model.output_shape == (None, 4, 4, 1024)

    #Upscale using Conv2DTranspose
    # ToDo: Look into upsampling via interpolation and 2d Convolution
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same',
                                     use_bias=False,
                                     kernel_initializer=weight_initializer))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same',
                                     use_bias=False,
                                     kernel_initializer=weight_initializer))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same',
                                     use_bias=False,
                                     kernel_initializer=weight_initializer))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same',
                                     use_bias=False, activation='tanh',
                                     kernel_initializer=weight_initializer))
    assert model.output_shape == (None, 64, 64, 3)
    return model

if __name__ == "__main__":
    generator = make_generator_model()
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    generated_image = (generated_image.numpy()*127.5 + 127.5)/255.0
    # generated_image = generated_image.astype(int)
    print(generated_image.shape)
    import matplotlib.pyplot as plt
    print(generated_image.min(), generated_image.max())
    fig, ax = plt.subplots()
    ax.imshow(generated_image[0, :, :, :])
    plt.show()