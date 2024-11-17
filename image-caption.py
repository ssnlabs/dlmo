import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import kagglehub
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
import tensorflow as tf

path = kagglehub.dataset_download("adityajn105/flickr8k")
caption_file = os.path.join(path, 'captions.txt')
image_folder = os.path.join(path, 'Images')

def load_captions(caption_file, num_images=20):
    df = pd.read_csv(caption_file)    
    image_ids = df['image'].unique()[:num_images]
    df = df[df['image'].isin(image_ids)]
    
    captions_dict = {}
    for _, row in df.iterrows():
        if row['image'] not in captions_dict:
            captions_dict[row['image']] = []
        caption = 'startseq ' + row['caption'].lower() + ' endseq'
        captions_dict[row['image']].append(caption)
    
    return captions_dict

captions_dict = load_captions(caption_file)

def load_images(image_folder, image_names):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    features = {}
    for img_name in image_names:
        img_path = os.path.join(image_folder, img_name)
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = tf.keras.applications.vgg16.preprocess_input(img)
        
        feature = model.predict(img, verbose=0)
        features[img_name] = feature
    
    return features

features = load_images(image_folder, list(captions_dict.keys()))

def prepare_text_data(captions_dict):
    all_captions = []
    for img_captions in captions_dict.values():
        all_captions.extend(img_captions)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    
    max_length = max(len(caption.split()) for caption in all_captions)
    
    return tokenizer, vocab_size, max_length

tokenizer, vocab_size, max_length = prepare_text_data(captions_dict)

def prepare_training_data(captions_dict, features, tokenizer, max_length, vocab_size):
    X1, X2, y = [], [], []
    
    for img_name, captions in captions_dict.items():
        feature = features[img_name][0]
        
        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]
            
            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]
                
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)
    
    return np.array(X1), np.array(X2), np.array(y)

X1, X2, y = prepare_training_data(captions_dict, features, tokenizer, max_length, vocab_size)

def create_tf_dataset(X1, X2, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(((X1, X2), y))
    dataset = dataset.shuffle(1000).batch(batch_size)
    return dataset

dataset = create_tf_dataset(X1, X2, y)

def create_model(vocab_size, max_length):    
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

model = create_model(vocab_size, max_length)

model.summary()

model.fit(dataset, epochs=20, verbose=1)

def generate_caption(model, tokenizer, feature, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ''
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break    
    caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return caption

test_image = list(features.keys())[7]
test_feature = features[test_image]
caption = generate_caption(model, tokenizer, test_feature, max_length)

img_path = os.path.join(image_folder, test_image)
img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.show()

print(f"Generated caption for {test_image}: {caption}")
