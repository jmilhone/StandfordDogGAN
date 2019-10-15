import tensorflow as tf
from discriminator import make_discriminator_model
from generator import make_generator_model
import os
import data_preparation
import loss_functions
import matplotlib.pyplot as plt


noise_dim = 100  #size input for the generator

@tf.function
def train_step(images, batch_size, generator, discriminator, generator_loss,
               discriminator_loss, generator_opt, discriminator_opt,
               use_smoothing=True, use_noise=True):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_image_noise = images + tf.random.normal(images.shape,
                  stddev=tf.random.uniform([1], minval=0.0, maxval=0.1))
        # real_output = discriminator(images, training=True)
        real_output = discriminator(real_image_noise, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output, loss_functions.gen_loss_gan,
                                  use_smoothing=use_smoothing)
        disc_loss = discriminator_loss(real_output, fake_output,
                                       loss_functions.disc_loss_gan,
                                       use_smoothing=use_smoothing,
                                       use_noise=use_noise,
                                       )
    grad_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    grad_of_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_opt.apply_gradients(zip(grad_of_gen, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(grad_of_disc, discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(dataset, epochs, start_epoch, batch_size, generator, discriminator, generator_loss,
          discriminator_loss, generator_opt, discriminator_opt, checkpoint, manager,
          image_folder, test_input, use_smoothing=True, use_noise=True):

    for epoch in range(start_epoch, start_epoch+epochs):
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, batch_size, generator, discriminator,
                                             generator_loss, discriminator_loss, generator_opt,
                                             discriminator_opt, use_smoothing=use_smoothing,
                                             use_noise=use_noise)
            checkpoint.step.assign_add(1)

            if int(checkpoint.step) % 100 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
                print(f"Gen Loss: {gen_loss.numpy():1.3e}  Disc. Loss: {disc_loss.numpy():1.3e}")

        images = generate_and_save_images(generator, epoch, test_input, image_folder)
        # print(tf.math.sigmoid(discriminator(images)).numpy().reshape((4,4)))
        print(discriminator(images).numpy().reshape((4,4)))


def generate_and_save_images(model, epoch, test_input, output_folder):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8, 8))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :, :]*127.5+127.5)/255.0)
        plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"image_at_epoch_{epoch:04d}.png"))
    plt.close(fig)

    return predictions

def train_gan(data, checkpoint_dir, start_epoch=100,  epochs=200, restart=False,
              batch_size=64, lr_gen=2e-4, lr_disc=2e-4, num_examples_to_generate=16):

    # Create the generator and discriminator
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    generator_loss = loss_functions.generator_loss
    discriminator_loss = loss_functions.discriminator_loss

    generator_optimizer = tf.keras.optimizers.Adam(lr_gen, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr_disc, beta_1=0.5)

    seed = tf.random.normal([num_examples_to_generate, noise_dim], seed=42)

    image_folder = os.path.join(checkpoint_dir, "Images")
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
        if not os.path.isdir(image_folder):
            print(f"Unable to create {image_folder}. Exiting train_gan.")
            return

    # checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                             generator_optimizer=generator_optimizer,
                             discriminator_optimizer=discriminator_optimizer,
                             generator=generator,
                             discriminator=discriminator,
                         )
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_folder, max_to_keep=5)
    if restart:
        print("Initializing from scratch.")
    else:
        checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

    train(data, epochs, start_epoch, batch_size, generator, discriminator, generator_loss,
          discriminator_loss, generator_optimizer, discriminator_optimizer, checkpoint, manager, image_folder,
          seed, use_smoothing=True, use_noise=True)


def flip(image):
    return tf.image.random_flip_left_right(image)


if __name__ == "__main__":
    buffer_size = 5000
    batch_size = 32

    dog_info = data_preparation.parse_all_annotations("Annotation")
    dog_images = data_preparation.prep_all_images(dog_info, final_image_size=64)
    print(dog_images.shape)
    dog_images_modified = (dog_images - 127.5) / 127.5
    dog_images_modified = dog_images_modified.astype('float32')
    dog_dataset = (tf.data.Dataset.from_tensor_slices(dog_images_modified)
                   .shuffle(buffer_size).map(flip).batch(batch_size).prefetch(5)
                   )


    checkpoint_folder = "Checkpoints/2019_10_14_1"
    train_gan(dog_dataset, checkpoint_folder, restart=False, batch_size=batch_size, lr_disc=2e-4, lr_gen=2e-4)

    # 2019_10_09_0 batchsize=64, lr = 2e-4
    # 2019_10_10_0 batchsize=128 lr_disc = 1e-4, lr_gen=4e-4
    # 2019_10_12_0 batchsize=128 lr_disc = 1e-4, lr_gen=4e-4, random_left_right_flip,
    #   kernel_initializer
    # 2019_10_12_1 batchsize=128 lr_disc = 1e-4, lr_gen=4e-4, random_left_right_flip,
    #   kernel_initializer, was able to fix the use noise function
    # 2019_10_12_2 batchsize=256 lr_disc = 2e-4, lr_gen=2e-4, random_left_right_flip,
    #   kernel_initializer, was able to fix the use noise function
    # 2019_10_13_0 I think I was missing an activation in the discrimintator at the last convolution before flatten
    # 2019_10_13_1 Moved batch normalization to after LeakyReLU
    # 2019_10_14_0 Add noise to real images, only have images with exactly one dog in them
    # 2019_10_14_0 Using sigmoid instead of logits

    # ToDo: Try making real 0 and fake 1. Go back to symmetric learning rates. Try implementing spectral normalization.
    # ToDo: Try adding noise to the actual images before feeding to the discriminator.
