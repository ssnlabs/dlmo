!pip install numpy
!pip install opencv-python
!pip install matplotlib
!pip install seaborn
!pip install tensorflow
!pip install scikit-learn


import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train = x_train[:10000], y_train[:10000]  
x_test, y_test = x_test[:2000], y_test[:2000]        

target_size = (32, 32)
x_train = np.array([cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), target_size) for img in x_train])
x_test = np.array([cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), target_size) for img in x_test])

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

def visualize_samples(images, labels):
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {np.argmax(labels[i])}")
        plt.axis('off')
    plt.show()

visualize_samples(x_train[:6], y_train[:6])

def create_model(base_model, num_classes=10, learning_rate=0.001):
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

input_shape = (*target_size, 3)

base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
for layer in base_model_resnet.layers:
    layer.trainable = False  
model_resnet = create_model(base_model_resnet)

base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
for layer in base_model_vgg16.layers:
    layer.trainable = False  
model_vgg16 = create_model(base_model_vgg16)

print("Training ResNet50 model...")
history_resnet = model_resnet.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=32
)

print("Training VGG16 model...")
history_vgg16 = model_vgg16.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=5,
    batch_size=32
)

def evaluate_model(model, x_test, y_test, name="Model"):
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"\n{name} Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")    
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{name} - Confusion Matrix')
    plt.show()

evaluate_model(model_resnet, x_test, y_test, "ResNet50")
evaluate_model(model_vgg16, x_test, y_test, "VGG16")