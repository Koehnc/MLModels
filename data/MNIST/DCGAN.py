import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, img_shape):
        super(Generator, self).__init__()
        
        self.input_dim = input_dim
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 128, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, img_shape, kernel_size=2, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), self.input_dim, 1, 1)
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.leaky_relu(self.conv1(x), 0.2)
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.leaky_relu(self.conv2(x), 0.2)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32*7*7)
        x = nn.functional.leaky_relu(self.fc1(x), 0.2)
        x = nn.functional.sigmoid(self.fc2(x))
        return x