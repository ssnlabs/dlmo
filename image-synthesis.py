import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image

def load_data(image_folder, caption_file, limit=20):
    imgs, image_paths, captions = [], [], []
    unique_images = set()
    with open(caption_file, 'r') as file:
        lines = file.readlines()
    for line in lines:
        img_name, caption = line.strip().split(',', 1)
        img_path = os.path.join(image_folder, img_name)
        if len(unique_images) >= limit:
            break
        if img_name not in unique_images and os.path.exists(img_path):
            unique_images.add(img_name)
            imgs.append(img_name)
            image_paths.append(img_path)
            captions.append(caption)
    return imgs, image_paths, captions

image_folder = r"Images"
caption_file = r"captions.txt"

imgs, image_paths, captions = load_data(image_folder, caption_file, limit=20)

def display_images(image_paths, limit=10):
    plt.figure(figsize=(10, 10))
    for i in range(min(limit, len(image_paths))):
        img = Image.open(image_paths[i]).resize((64, 64))
        plt.subplot(5, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

display_images(image_paths)


def preprocess_images(image_paths):
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).resize((64, 64))
        img = np.array(img) / 127.5 - 1.0  
        images.append(img)
    return np.array(images)

images = preprocess_images(image_paths)


def build_generator():
    noise_input = layers.Input(shape=(100,))
    text_input = layers.Input(shape=(100,))
    combined_input = layers.Concatenate()([noise_input, text_input])

    x = layers.Dense(8 * 8 * 256, activation="relu")(combined_input)
    x = layers.Reshape((8, 8, 256))(x)

    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu")(x)
    output = layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", activation="tanh")(x)

    return Model([noise_input, text_input], output, name="Generator")


def build_discriminator():
    image_input = layers.Input(shape=(64, 64, 3))
    text_input = layers.Input(shape=(100,))

    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same", activation="leaky_relu")(image_input)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same", activation="leaky_relu")(x)
    x = layers.Flatten()(x)

    combined_input = layers.Concatenate()([x, text_input])
    x = layers.Dense(256, activation="leaky_relu")(combined_input)
    output = layers.Dense(1, activation="sigmoid")(x)

    return Model([image_input, text_input], output, name="Discriminator")


generator = build_generator()
discriminator = build_discriminator()

generator.summary()
discriminator.summary()
