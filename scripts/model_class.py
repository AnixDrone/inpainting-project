import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models

def initialize_model(output_size):
    model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, output_size)
    return model

class InpaintingVAE(nn.Module):
    def __init__(self,
                 in_channels,
                 latent_dim,
                 device,
                 features_d = 64):
        super().__init__()
        
        self.device = device
        self.latent_dim = latent_dim
        self.input_channel = in_channels
        
        #self.embed_class = nn.Embedding(num_classes, self.image_size * self.image_size)
        #self.embed_data = nn.Conv2d(in_channels, in_channels,1)
        
        
            
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(in_channels, features_d, 4, 2, 1), # 64 -> 32
        #     nn.InstanceNorm2d(features_d, affine=True),
        #     nn.LeakyReLU(0.2),
        #     self._encoder_block(features_d, features_d * 2, 4, 2, 1),# 32 -> 16
        #     self._encoder_block(features_d * 2, features_d * 4, 4, 2, 1),# 16 - > 8
        #     self._encoder_block(features_d * 4, features_d * 8, 4, 2, 1), # 8 -> 4
        # )

        #self.fc_mu = nn.Linear(features_d * 8 * 2 * 2, latent_dim)
        #self.fc_var = nn.Linear(features_d * 8 * 2 * 2, latent_dim)
        self.encoder = initialize_model(latent_dim)
        
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        
        
        self.decoder_input = nn.Linear(latent_dim, 131072)
        
        self.decoder = nn.Sequential(
            self._decoder_bloc(features_d * 8, features_d * 4, 4, 2, 1),# 32 -> 16
            self._decoder_bloc(features_d * 4, features_d * 2, 4, 2, 1),# 16 - > 8
            self._decoder_bloc(features_d * 2, features_d, 4, 2, 1), # 8 -> 4
            nn.ConvTranspose2d(features_d, in_channels, 4, 2, 1), # 64 -> 32
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 1), # 32 -> 64
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        
    def _encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def _decoder_bloc(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def encode(self,input):
        result = self.encoder(input)
        #print(result.shape)
        #result = result.flatten(1)
        
        if torch.isnan(result).any().item():
            print(result)
        
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return mu, log_var
    
    def decode(self,z):
        result  = self.decoder_input(z)
        #print(result.shape)
        result  = result.view(-1,512,16,16)
        #print(result.shape)
        result = self.decoder(result)
        return result
    
    def reparameterize(self, mu, log_var):
        
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        
    
    def forward(self,input):
        
        x = input
        
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        
        
        return self.decode(z), mu, log_var
    
    def sample(self, n,label):
        label = label.unsqueeze(1)
        if self.device is not None:
            z = torch.randn(n, self.latent_dim).to(self.device)
        else:
            z = torch.randn(n, self.latent_dim)
        z = torch.cat([z,label],dim=1)
        #z = z.to(self.device)
        
        return self.decode(z)