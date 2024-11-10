import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

base_model = VGG16(weights='imagenet', include_top=False)
img_path = 'horse.jpg' 

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Scale to [0, 1]
features = base_model.predict(img_array)

print("Feature map shape:", features.shape)

plt.figure(figsize=(12, 12))
for i in range(6):  
    plt.subplot(2, 3, i + 1)
    plt.imshow(features[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.suptitle('Feature Maps from VGG16')
plt.show()
