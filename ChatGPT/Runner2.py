import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from DCGAN import Generator, Discriminator

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_size = 100
batch_size = 128
num_epochs = 50
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # normalize to [-1, 1]
])
dataset = MNIST(root='data/', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize generator and discriminator
generator = Generator(latent_size).to(device)
discriminator = Discriminator().to(device)

# Define loss function and optimizers
criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
dis_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Define fixed noise for evaluation during training
fixed_noise = torch.randn(batch_size, latent_size, 1, 1, device=device)

# Training loop
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Train discriminator
        dis_optimizer.zero_grad()
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        real_preds = discriminator(real_images)
        dis_loss_real = criterion(real_preds, real_labels)
        z = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = generator(z)
        fake_preds = discriminator(fake_images.detach())
        dis_loss_fake = criterion(fake_preds, fake_labels)
        dis_loss = dis_loss_real + dis_loss_fake
        dis_loss.backward()
        dis_optimizer.step()

        # Train generator
        gen_optimizer.zero_grad()
        z = torch.randn(batch_size, latent_size, 1, 1, device=device)
        fake_images = generator(z)
        fake_preds = discriminator(fake_images)
        gen_loss = criterion(fake_preds, real_labels)
        gen_loss.backward()
        gen_optimizer.step()

        # Print losses
        if i % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Batch {i}/{len(dataloader)} | "
                  f"Discriminator loss: {dis_loss.item():.4f} | Generator loss: {gen_loss.item():.4f}")

    # Save generated images
    with torch.no_grad():
        fake_images = generator(fixed_noise)
        fake_images = (fake_images + 1) / 2 # unnormalize to [0, 1]
        save_image(fake_images, f"images/fake_{epoch+1:03d}.png", nrow=10)

    # Save models
    torch.save(generator.state_dict(), f"models/generator_{epoch+1:03d}.pt")
    torch.save(discriminator.state_dict(), f"models/discriminator_{epoch+1:03d}.pt")