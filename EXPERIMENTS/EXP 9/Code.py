# --------------------------------------------------------
# AIM: To build a Generative Adversarial Network (GAN) using Keras/TensorFlow
# --------------------------------------------------------

# Step 1: Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load and preprocess the dataset (MNIST)
(X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28*28)  # Flatten images to 784-dim vectors

# Step 3: Define the generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.Dense(784, activation='sigmoid')
    ])
    return model

# Step 4: Define the discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create generator and discriminator
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Build the combined GAN model
discriminator.trainable = False
gan_input = layers.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Step 6: Train the GAN
epochs = 1000
batch_size = 128
half_batch = batch_size // 2

for epoch in range(epochs):
    # Select a random half batch of real images
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    real_images = X_train[idx]
    
    # Generate a half batch of fake images
    noise = np.random.randn(half_batch, 100)
    fake_images = generator.predict(noise)
    
    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train the generator
    noise = np.random.randn(batch_size, 100)
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f} | D Acc: {d_loss[1]*100:.2f}% | G Loss: {g_loss:.4f}")

# Step 7: Generate new images after training
noise = np.random.randn(16, 100)
generated_images = generator.predict(noise)
generated_images = generated_images.reshape(16, 28, 28)

# Step 8: Visualize generated images
plt.figure(figsize=(6,6))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(generated_images[i], cmap='gray')
    plt.axis('off')
plt.suptitle("Generated Digits by GAN", fontsize=14)
plt.show()
