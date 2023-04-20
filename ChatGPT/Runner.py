import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from dcgan import DCGAN

# Load the MNIST dataset
(X_train, _), (_, _) = mnist.load_data()

# Normalize the pixel values to be between -1 and 1
X_train = (X_train.astype('float32') - 127.5) / 127.5

# Reshape the images to be 28x28x1 and add a channel dimension
X_train = np.expand_dims(X_train, axis=-1)

# Create the DCGAN model
dcgan = DCGAN(image_shape=(28, 28, 1))

# Compile the model
dcgan.compile()

# Train the model
dcgan.fit(X_train, epochs=100, batch_size=128)

# Generate some images from the model
noise = np.random.normal(size=(25, 100))
generated_images = dcgan.generator.predict(noise)

# Rescale the pixel values to be between 0 and 1
generated_images = (generated_images + 1) / 2

# Plot the generated images
fig, axs = plt.subplots(5, 5)
count = 0
for i in range(5):
    for j in range(5):
        axs[i,j].imshow(generated_images[count, :, :, 0], cmap='gray')
        axs[i,j].axis('off')
        count += 1
plt.show()