import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import confusion_matrix

# Step 1: Load and Split Dataset
data_dir = 'horse-or-human'  # Replace with actual dataset path
img_size = (128, 128)  # Resize images to 128x128

# Use ImageDataGenerator for loading and splitting
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 80-20 train-validation split

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

# Step 2: Visualize Samples
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

# Step 3: Load and Fine-Tune Pre-trained Models
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

# Load MobileNetV2 and VGG16
base_model_mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(*img_size, 3))

# Freeze base model layers for feature extraction
for layer in base_model_mobilenet.layers:
    layer.trainable = False
for layer in base_model_vgg16.layers:
    layer.trainable = False

# Initialize models with the base networks
model_mobilenet = create_model(base_model_mobilenet, img_size)
model_vgg16 = create_model(base_model_vgg16, img_size)

# Step 4: Train and Evaluate Models with Different Hyperparameters
history_mobilenet = model_mobilenet.fit(
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

# Step 5: Evaluate and Report Results
def evaluate_model(model, generator, name="Model"):
    loss, accuracy = model.evaluate(generator)
    print(f"\n{name} Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
    y_pred = (model.predict(generator) > 0.5).astype(int)
    y_true = generator.classes
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Horse', 'Human'], yticklabels=['Horse', 'Human'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{name} - Confusion Matrix')
    plt.show()

# Evaluate both models on validation data
evaluate_model(model_mobilenet, validation_generator, "MobileNetV2")
evaluate_model(model_vgg16, validation_generator, "VGG16")
