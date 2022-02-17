"""

Model: AlphaGAN
Programmer: Muhammad Hanan Asghar
Date: 10-02-2022
Day: Thursday
Language: Python_3.9
Library: Tensorflow_2.x
Github: https://github.com/MuhammadHananAsghar

"""
from utils.save import save_images
from utils.loss import generator_loss, discriminator_loss
from models.generator import build_generator
from models.discriminator import build_discriminator
from alphagenerator.alphagenerator import AlphaDatagen
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import os

# Generator Parameters
SIZE = 32  # 32, 64, 128, 256
SEED = 100  # 100, 200, 300


# Discriminator
IMAGE_SHAPE = (SIZE, SIZE, 3)


# Directories
IMAGES = "/content/images"
IMAGES_PATH = "/content/images"
OUTPUT_PATH = "/content/drive/MyDrive/AlphaGAN/output"
MODEL_PATH = '/content/drive/MyDrive/AlphaGAN/model'
BATCH_SIZE = 96

# Loss Functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

# Gradients Changing Function


@tf.function
def train_step(images):
    seed = tf.random.normal([BATCH_SIZE, SEED])
    # GradientTape make generator and discriminator trains together
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate Fake Image from generator
        generated_images = generator(seed, training=True)
        # Training real dataset images in discriminator
        real_output = discriminator(images, training=True)
        # Training generator generated images in discriminator
        fake_output = discriminator(generated_images, training=True)
        # Calculate Generator loss
        gen_loss = generator_loss(cross_entropy, fake_output)
        # Calculate Discriminator loss
        disc_loss = discriminator_loss(cross_entropy, real_output, fake_output)

        # Getting Generator Model Gradients
        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables)
        # Getting Discriminator Model Gradients
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)

        # Updating Generator Model Gradients
        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables))

        # Updating Discriminator Model Gradients
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss, real_output, fake_output


# Models
generator = build_generator(SEED_SIZE=SEED, size=SIZE)
discriminator = build_discriminator(image_shape=IMAGE_SHAPE, size=SIZE)

# AlhpaDataGenerator
Images = os.listdir(IMAGES)
datagenerator = AlphaDatagen(
    Images, IMAGES_PATH, batch_size=BATCH_SIZE, input_size=IMAGE_SHAPE)


# MAIN FUNCTION
def train(dataset, epochs):
    PREVIEW_ROWS = 4
    PREVIEW_COLS = 7
    fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, SEED))
    # Start
    for epoch in range(epochs):
        # Defining Variables
        gen_loss_list = []
        disc_loss_list = []
        real_list = []
        fake_list = []
        # Number of batches
        num_batches = dataset.__len__()
        # Iterating through images
        for _ in tqdm(range(0, num_batches), total=num_batches,
                      desc=f"[*] Epoch=> {epoch+1}"):
            images_batch = next(dataset)
            t = train_step(images_batch)
            # Saving Data of current batch
            gen_loss_list.append(t[0])
            disc_loss_list.append(t[1])

            real_out = t[2].numpy()
            fake_out = t[3].numpy()

            # If discriminator Real Image output is greater than 0.5 then == 1.
            # If discrimintor Real Image output is less than and equal to 0.5 then == 0.
            real_out[real_out > .5] = 1
            real_out[real_out <= .5] = 0

            # Height and Width of real image
            height, width = real_out.shape[:2]
            real_right = sum(real_out) / (height * width)

            real_list.append(real_right)
            # If discriminator Fake Image output is greater than 0.5 then == 0.
            # If discrimintor Fake Image output is less than and equal to 0.5 then == 1.
            fake_out[fake_out > .5] = 0
            fake_out[fake_out <= .5] = 1

            height, width = fake_out.shape[:2]
            fake_right = sum(fake_out) / (height * width)

            fake_list.append(fake_right)

        # Calculating Losses and Accuracy Average
        g_loss = sum(gen_loss_list) / len(gen_loss_list)
        d_loss = sum(disc_loss_list) / len(disc_loss_list)
        r_acc = sum(real_list) / len(real_list)
        f_acc = sum(fake_list) / len(fake_list)

        print(f'Epoch {epoch+1}, Gen Loss = {g_loss}, Disc Loss = {d_loss}')
        # Saving Image
        save_images(generator, epoch+1, fixed_seed, OUTPUT_PATH, MODEL_PATH, SIZE, activation="tanh")

# Run
train(datagenerator, 1000)