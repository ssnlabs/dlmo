import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, RepeatVector
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
import cv2
import matplotlib.pyplot as pt
#https://www.kaggle.com/datasets/adityajn105/flickr8k
import os
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
image_paths = image_paths[1:12]
captions = captions[1:12]

cnn_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
cnn_model = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-1].output)

import matplotlib.pyplot as plt

def extract_features(image):
    image = np.expand_dims(image, axis=0)  
    return cnn_model.predict(image)

def build_captioning_model(vocab_size, max_caption_length):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=256, input_length=max_caption_length))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

vocab_size = 1000           
max_caption_length = 10      

captioning_model = build_captioning_model(vocab_size, max_caption_length)
captioning_model.summary()

image = np.random.rand(224, 224, 3)  
caption = np.random.randint(1, vocab_size, (1, max_caption_length))

image_features = extract_features(image)
print("Extracted Image Features:", image_features.shape)

captioning_model.fit(caption, np.random.rand(1, max_caption_length, vocab_size), epochs=10,verbose=1)

reference_captions = [
    ["a", "sample", "caption", "of", "an", "image"],
    ["another", "description", "of", "the", "image", "content"]
]

!pip install nltk

from nltk.translate.bleu_score import sentence_bleu

def evaluate_bleu(reference, candidate):
    reference = [reference]  
    return sentence_bleu(reference, candidate)

dummy_generated_caption = ["this", "is", "a", "generated", "caption"]
bleu_score = evaluate_bleu(reference_captions[0], dummy_generated_caption)
print("BLEU score for the generated caption:", bleu_score)