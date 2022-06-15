import glob
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import keras.backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# tensorflow 2.7. has backwards compatibility problems
tf.config.run_functions_eagerly(True)

# directories
input_directory = 'C:\\heightmaps\\use_these\\resized\\rivers'
output_directory = 'C:\\GAN\\3D-Terrain-Generation-GAN\\figs'
checkpoint_directory = 'C:\\GAN\\3D-Terrain-Generation-GAN\\training_checkpoints'
dataset_glob = list(glob.glob("C:\\heightmaps\\use_these\\resized\\coasts\\*.png"))

# Batch and target sizes
BATCH_SIZE = 32
TARGET_SIZE = (256, 256)
loss_list_1 = []
gen_losses = []
disc_losses = []


# normalizing input images
def change_range(image):
    return [(i / 128.0) - 1 for i in image[:, :, :]]


image_generator = ImageDataGenerator(preprocessing_function=change_range, horizontal_flip=True, vertical_flip=True)

# real grayscale images from the dataset
real_generator = image_generator.flow_from_directory(input_directory, batch_size=BATCH_SIZE, shuffle=True, target_size=TARGET_SIZE, color_mode="grayscale", class_mode='input')


def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Reshape(target_shape=[1, 1, 4096], input_shape=[4096]))

    model.add(layers.Conv2DTranspose(filters=256, kernel_size=4))
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Conv2D(filters=256, kernel_size=4, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.7))
    model.add(layers.UpSampling2D())

    model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.7))
    model.add(layers.UpSampling2D())

    model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.7))
    model.add(layers.UpSampling2D())

    model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.7))
    model.add(layers.UpSampling2D())

    model.add(layers.Conv2D(filters=16, kernel_size=3, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization(momentum=0.7))
    model.add(layers.UpSampling2D())

    model.add(layers.Conv2D(filters=8, kernel_size=3, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.UpSampling2D())

    model.add(layers.Conv2D(filters=1, kernel_size=3, padding='same'))
    model.add(layers.Activation('tanh'))

    return model


generator = generator_model()


def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.GaussianNoise(0.2, input_shape=[256, 256, 1]))

    # 256x256x3 Image
    model.add(layers.Conv2D(filters=8, kernel_size=3, padding='same'))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.AveragePooling2D())

    # 128x128x8
    model.add(layers.Conv2D(filters=16, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.7))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.AveragePooling2D())

    # 64x64x16
    model.add(layers.Conv2D(filters=32, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.7))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.AveragePooling2D())

    # 32x32x32
    model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.7))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.AveragePooling2D())

    # 16x16x64
    model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.7))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.AveragePooling2D())

    # 8x8x128
    model.add(layers.Conv2D(filters=256, kernel_size=3, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.7))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.25))
    model.add(layers.AveragePooling2D())

    # 4x4x256
    model.add(layers.Flatten())

    # 256
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(0.2))

    model.add(layers.Dense(1, activation='sigmoid'))

    return model


discriminator = discriminator_model()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_values = np.random.uniform(0.95, 1.0, size=real_output.get_shape())
    fake_values = np.random.uniform(0.0, 0.05, size=fake_output.get_shape())

    real_loss = cross_entropy(real_values, real_output)
    fake_loss = cross_entropy(fake_values, fake_output)

    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# optimizer values
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
generator_optimizer = tf.keras.optimizers.Adam(1e-5)


# save checkpoints
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# training values
noise = 4096
parallel_image_count = 10
seed = tf.random.normal([parallel_image_count, noise])


@tf.function
def train_step(images):
    randomized_noise = tf.random.normal([BATCH_SIZE, noise])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_img_output = generator(randomized_noise, training=True)

        real_output = discriminator(images[0], training=True)
        fake_output = discriminator(gen_img_output, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gen_losses.append(keras.backend.eval(gen_loss))
        disc_losses.append(keras.backend.eval(disc_loss))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


def main_training(dataset, start_epoch, epochs):
    for epoch in range(start_epoch, start_epoch + epochs):
        start_time = time.time()

        batches = 0
        for image_batch in dataset:
            train_step(image_batch)

            batches += 1
            if batches >= len(dataset_glob) / BATCH_SIZE:
                break

        image_saver(generator, epoch + 1, seed)

        # save checkpoints every 100 epochs, clogs memory on excessive saving.
        if (epoch + 1) % 100 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('epoch count {} done at {} secs'.format(epoch + 1, time.time() - start_time))

    # plot out the loss graphs
    plt.plot(gen_losses, label="generator loss")
    plt.plot(disc_losses, label="discriminator loss")
    plt.legend()
    plt.show()

    image_saver(generator, epochs, seed)


def image_saver(model, epoch, test_input):
    predictions = model(test_input, training=False)

    for i in range(predictions.shape[0]):
        plt.imsave(output_directory + '\\output_of_epoch_{:04d}'.format(epoch) + '_ex_' + str(i) + '.png',
                   predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')


# intended epoch count to train
checkpoint_epoch = 0
goal_epoch_count = 1000

# start training
main_training(real_generator, checkpoint_epoch, goal_epoch_count)
