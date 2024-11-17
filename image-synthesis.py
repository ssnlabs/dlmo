import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_dim=latent_dim))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(28*28, activation='tanh'))
    model.add(layers.Reshape((28, 28,1)))
    return model

def build_discriminator(input_shape):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False  
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

latent_dim = 100
image_shape = (28, 28)

discriminator = build_discriminator(image_shape)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

generator = build_generator(latent_dim)

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

batch_size = 64
epochs = 10
sample_interval = 5  

def train_gan(epochs, batch_size, sample_interval):
    half_batch = batch_size // 2

    for epoch in range(epochs):
        
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        real_images = x_train[idx]
        fake_images = generator.predict(np.random.randn(half_batch, latent_dim))

        
        real_labels = np.ones((half_batch, 1))
        fake_labels = np.zeros((half_batch, 1))

        
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        
        noise = np.random.randn(batch_size, latent_dim)
        valid_labels = np.ones((batch_size, 1))  
        g_loss = gan.train_on_batch(noise, valid_labels)

        
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
            save_generated_images(epoch)

def save_generated_images(epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.randn(examples, latent_dim)
    generated_images = generator.predict(noise)
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"gan_generated_image_{epoch}.png")
    plt.close()

def generate_new_images(num_images=10):
    noise = np.random.randn(num_images, latent_dim)  
    generated_images = generator.predict(noise)
    
    
    generated_images = (generated_images + 1) / 2.0
    
    
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(5, 5, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

train_gan(epochs, batch_size, sample_interval)
generate_new_images(num_images=10)
