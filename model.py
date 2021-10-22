import torch.nn as nn
import torch
import utils

# input: 256x256x3 sized image, output: 100x1x1 image feature
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__() 
        self.model = nn.Sequential(
            # State (3x256x256)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (16x128x128)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (32x64x64)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (64x32x32)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (128x16x16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x8x8)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x4x4)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=1, padding=0)
            # output of main module --> State (1024x1x1)
        )
        self.last_layer = nn.Sequential(
            #latent_dim = 100
            nn.Linear(1024, latent_dim),
            nn.Linear(latent_dim, latent_dim)
        )
        self.fc_mu = nn.Linear(in_features = 1024, out_features = latent_dim)
        self.fc_logvar = nn.Linear(in_features = 1024, out_features = latent_dim)

    def forward(self, img):
        features = self.model(img)
        features = features.view(img.shape[0], -1)
        x_mu = self.fc_mu(features)
        x_logvar = self.fc_logvar(features)
        return x_mu, x_logvar

#input:100x1x1 image features, output: 256x256x3 images
class Decoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Decoder, self).__init__() 
        self.img_shape = img_shape
        self.model = nn.Sequential(
            #Output = (Input-1) \times Stride - 2 \times Padding + Filter + OutputPadding 

            # State (100x1x1)
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(True),

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(True),

            # State (64x64x64)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(True),

            # State (32x128x128)
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid())

    def forward(self, input):
        img = self.model(input)
        return img.view(img.shape[0], *self.img_shape)


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__() 
        """Initialize the VAE model"""
        img_size = (3, 256, 256)
        latent = 100

        self.encoder = Encoder(latent)
        self.decoder = Decoder(img_size, latent)

    def latent_sample(self, mu, logvar):
        if self.training == True:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):            
        latent_mu, latent_logvar = self.encoder(x)       
        features = self.latent_sample(latent_mu, latent_logvar)
        features = features.view(features.shape[0], -1, 1, 1)
        g = self.decoder(features)
        return g, latent_mu, latent_logvar

