import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
import cv2
import matplotlib.pyplot as pt
import os

#https://www.kaggle.com/datasets/adityajn105/flickr8k
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

imgs, image_paths, captions = load_data(image_folder, caption_file, limit=500)

def build_generator():
    noise_input = layers.Input(shape=(100,))
    text_input = layers.Input(shape=(100,))
    combined = layers.Concatenate()([noise_input, text_input])

    model = Sequential()
    model.add(layers.Dense(8 * 8 * 128, activation="relu"))
    model.add(layers.Conv2DTranspose(3, 4, strides=4, activation="tanh"))
    return model

def build_discriminator():
    image_input = layers.Input(shape=(100,))
    text_input = layers.Input(shape=(100,))
    combined = layers.Concatenate()([image_input, text_input])
    model = Sequential()
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

generator = build_generator()
discriminator = build_discriminator()

generator.summary()
discriminator.summary()

import matplotlib.pyplot as plt
from PIL import Image

def display_images(image_paths, limit=20):
    plt.figure(figsize=(10, 10))    
    for i in range(min(limit, len(image_paths))):
        image = Image.open(image_paths[i])
        plt.subplot(5, 5, i + 1)  
        plt.imshow(image)
        plt.axis('off')
    plt.show()

display_images(image_paths)