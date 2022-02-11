import os
import numpy as np
from PIL import Image


def save_images(generator, cnt, noise, OUTPUT, MODEL, size, activation="tanh"):

    if activation == "tanh":
        
        PREVIEW_ROWS = 4
        PREVIEW_COLS = 7
        PREVIEW_MARGIN = 16
        IMAGE_SIZE = size
        CHANNELS = 3
        image_array = np.full((
            PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE+PREVIEW_MARGIN)),
            PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE+PREVIEW_MARGIN)), 3),
            255, dtype=np.uint8)

        generated_images = generator.predict(noise)
        generated_images = (generated_images + 1) / 2.0

        image_count = 0
        for row in range(PREVIEW_ROWS):
            for col in range(PREVIEW_COLS):
                r = row * (IMAGE_SIZE+16) + PREVIEW_MARGIN
                c = col * (IMAGE_SIZE+16) + PREVIEW_MARGIN
                image_array[r:r+IMAGE_SIZE, c:c +
                            IMAGE_SIZE] = generated_images[image_count] * 255
                image_count += 1

        filename = os.path.join(
            OUTPUT, f"train-{cnt}.png")
        im = Image.fromarray(image_array)
        im.save(filename)
        generator.save(os.path.join(
            MODEL, f"train-{cnt}.h5"))
        print("Saved Infrences at {}.".format(cnt))

    elif activation == "sigmoid":

        PREVIEW_ROWS = 4
        PREVIEW_COLS = 7
        PREVIEW_MARGIN = 16
        IMAGE_SIZE = size
        CHANNELS = 3
        image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE+PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE+PREVIEW_MARGIN)), 3),
        255, dtype=np.uint8)

        generated_images = generator.predict(noise)

        image_count = 0
        for row in range(PREVIEW_ROWS):
            for col in range(PREVIEW_COLS):
                r = row * (IMAGE_SIZE+16) + PREVIEW_MARGIN
                c = col * (IMAGE_SIZE+16) + PREVIEW_MARGIN
                image_array[r:r+IMAGE_SIZE, c:c +
                            IMAGE_SIZE] = generated_images[image_count] * 255
                image_count += 1

        filename = os.path.join(
            OUTPUT, f"train-{cnt}.png")
        im = Image.fromarray(image_array)
        im.save(filename)
        generator.save(os.path.join(
            MODEL, f"train-{cnt}.h5"))
        print("Saved Infrences at {}.".format(cnt))