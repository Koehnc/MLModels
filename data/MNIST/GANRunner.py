import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os

from DCGAN import *

# Load MNIST dataset
mnist = MNIST(root='data', train=True, download=True, transform=ToTensor())

# Choose one image from the dataset
image, label = mnist[0]
image = image.unsqueeze(0) # Add batch dimension


def save_sample_image(samples, save_path, nrow=16):
    # Normalize and clamp the samples to [0, 1] range
    samples = (samples + 1.0) / 2.0
    samples = torch.clamp(samples, 0, 1)
    

    # Create a grid of the samples and save as an image file
    grid = vutils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vutils.save_image(grid, save_path)

generator = Generator(100, 1)
discriminator = Discriminator()

# Define the loss functions and optimizers
criterion = nn.BCELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define the batch size and number of epochs
batch_size = 96
num_epochs = 50
disc_loss_y = []
gen_loss_y = []

# # Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

"""
"""

# Train the GAN
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        
        # Train the discriminator
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train the discriminator on real images
        real_output = discriminator(real_images)
        disc_real_loss = criterion(real_output, real_labels)

        # Train the discriminator on fake images
        z = torch.randn(batch_size, 100)
        fake_images = generator(z)
        fake_output = discriminator(fake_images.detach())
        disc_fake_loss = criterion(fake_output, fake_labels)

        # Compute the total discriminator loss and backpropagate
        disc_loss = disc_real_loss + disc_fake_loss
        disc_loss.backward()
        disc_optimizer.step()

        # Train the generator
        generator.zero_grad()
        z = torch.randn(batch_size, 100)
        fake_images = generator(z)
        fake_output = discriminator(fake_images)
        gen_loss = criterion(fake_output, real_labels)
        gen_loss.backward()
        gen_optimizer.step()

        # Append the loss to graph later
        disc_loss_y.append(disc_loss)
        gen_loss_y.append(gen_loss)

        # Print the loss for every 100th iteration
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                  f"Discriminator Loss: {disc_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}")

        # Save a sample image every 500th iteration
        if i % 500 == 0:
            filename = f"images/epoch{epoch}-gen{i}.png"
            print(fake_images.shape)
            save_path = os.path.join(os.getcwd(), filename)
            save_sample_image(fake_images, save_path)


plt.plot(disc_loss_y)
plt.plot(gen_loss_y)
plt.show()