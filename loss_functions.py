import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
# cross_entropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output, loss_func,
                       use_noise=False, use_smoothing=False):

    if not use_noise and not use_smoothing:
        return loss_func(real_output, fake_output)

    if use_noise and use_smoothing:
        # call both
        real_output_noise = noisy_labels(tf.ones_like(real_output), 0.05)
        fake_output_noise = noisy_labels(tf.zeros_like(fake_output), 0.05)
        real_output_smooth = smooth_positive_labels(real_output_noise)
        fake_output_smooth = smooth_negative_labels(fake_output_noise)
        # I think I've been doing this wrong...
        real_loss = cross_entropy(real_output_smooth, real_output)
        fake_loss = cross_entropy(fake_output_smooth, fake_output)
        return real_loss + fake_loss
        # return loss_func(real_output_smooth, fake_output_smooth)

    if use_noise and not use_smoothing:
        # call use_noise helper function
        real_output_noise = noisy_labels(tf.ones_like(real_output), 0.05)
        fake_output_noise = noisy_labels(tf.zeros_like(fake_output), 0.05)
        return loss_func(real_output_noise, fake_output_noise)

    # Last case is to call smoothing but not noise
    # I'm changing this because I think I'm doing it wrong...
    real_output_smooth = smooth_positive_labels(tf.ones_like(real_output))
    fake_output_smooth = smooth_negative_labels(tf.zeros_like(fake_output))
    real_loss = cross_entropy(real_output_smooth, real_output)
    fake_loss = cross_entropy(fake_output_smooth, fake_output)
    return real_loss + fake_loss
    # return loss_func(real_output_smooth, fake_output_smooth)


def generator_loss(fake_output, loss_func, use_smoothing=False):
    if use_smoothing:
        # call smoothing
        # I think I'm doing this wrong...
        fake_output_smoothing = smooth_positive_labels(tf.ones_like(fake_output))
        return cross_entropy(fake_output_smoothing, fake_output)
        # fake_output_smoothing = smooth_negative_labels(fake_output)
        # return loss_func(fake_output_smoothing)
    else:
        return loss_func(fake_output)


def disc_loss_gan(real_output, fake_output):
    """Helper function for discriminator loss. Calclates standard GAN loss"""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def gen_loss_gan(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def disc_loss_gan_reversed_label(real_output, fake_output):
    real_loss = cross_entropy(tf.zeros_like(real_output), real_output)
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return real_loss + fake_loss


def gen_loss_gan_reversed_label(fake_output):
    return cross_entropy(tf.zeros_like(fake_output), fake_output)


def smooth_positive_labels(y):
    return y - 0.3 * np.random.random(y.shape)


def smooth_negative_labels(y):
    return y + 0.3 * np.random.random(y.shape)


def noisy_labels(y, prob):

    npts = int(y.shape[0])
    nflips = int(prob * npts)

    indices = np.random.choice(range(npts), size=nflips)

    op_list = []
    for i in range(npts):
        if i in indices:
            op_list.append(tf.subtract(1.0, y[i]))
        else:
            op_list.append(y[i])
    outputs = tf.stack(op_list)
    return outputs


if __name__ == "__main__":
    pos_vals = np.ones(10000)
    pos_smooth = smooth_positive_labels(pos_vals)

    neg_vals = np.zeros(10000)
    neg_smooth = smooth_negative_labels(neg_vals)

    fig, ax = plt.subplots()
    ax.hist(pos_smooth, bins=30)
    ax.hist(neg_smooth, bins=30)
    plt.show()
