import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Define the path to your dataset directory
dataset_dir = 'horse-or-human'  # Replace with the actual path

# Load the dataset with a train-validation split
full_data = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary',
    validation_split=0.3,  # Use 30% for validation
    subset="training",
    seed=42
)

# Further split validation into validation and test sets
val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(128, 128),
    batch_size=32,
    label_mode='binary',
    validation_split=0.3,  # Ensure the same 30% split
    subset="validation",
    seed=42
)

# Split val_data into 50-50 for validation and test sets manually
val_batches = tf.data.experimental.cardinality(val_data)
test_data = val_data.take(val_batches // 2)  # Take 50% for testing
val_data = val_data.skip(val_batches // 2)  # Remaining 50% for validation

# Normalize the data (scale pixel values to [0, 1])
normalization_layer = layers.Rescaling(1.0 / 255)
train_data = full_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))
test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

# Define the class names for the binary labels
class_names = ['Horse', 'Human']

# Visualize some samples from the dataset
def plot_samples(dataset, num_samples=6):
    plt.figure(figsize=(12, 6))
    for i, (images, labels) in enumerate(dataset.take(num_samples)):
        plt.subplot(2, 3, i + 1)    
        plt.imshow(images[0].numpy())
        label = int(labels[0].numpy())  # Convert label to int for lookup
        plt.title(f"Label: {class_names[label]}")
        plt.axis('off')
    plt.show()

plot_samples(train_data)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data)
print(f"\nTest accuracy: {test_acc}")

# Plot training and validation accuracy over epochs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Generate confusion matrix on the test set
y_pred = np.round(model.predict(test_data).flatten())
y_true = np.concatenate([label.numpy() for _, label in test_data], axis=0)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
