import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import confusion_matrix

data_dir = 'horse-or-human'
img_size = (128, 128)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

def visualize_samples(generator):
    images, labels = next(generator)
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i])
        plt.title(f"Label: {'Horse' if labels[i] == 0 else 'Human'}")
        plt.axis('off')
    plt.show()

visualize_samples(train_generator)

def create_model(base_model, img_size, learning_rate=0.001):
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(*img_size, 3))

for layer in base_model_resnet.layers:
    layer.trainable = False

model_resnet = create_model(base_model_resnet, img_size)

base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(*img_size, 3))

for layer in base_model_vgg16.layers:
    layer.trainable = False

model_vgg16 = create_model(base_model_vgg16, img_size)

history_resnet = model_resnet.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    batch_size=32
)

history_vgg16 = model_vgg16.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5,
    batch_size=32
)

def evaluate_model(model, generator, name="Model"):
    loss, accuracy = model.evaluate(generator)
    print(f"\n{name} Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
    y_pred = (model.predict(generator) > 0.5).astype(int)
    y_true = generator.classes
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Horse', 'Human'], yticklabels=['Horse', 'Human'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{name} - Confusion Matrix')
    plt.show()

evaluate_model(model_resnet, validation_generator, "ResNet50")
evaluate_model(model_vgg16, validation_generator, "VGG16")
