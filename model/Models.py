import torch
import torch.nn as nn
from torchsummary import summary
from Denormalization import DenormResBlock
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import spectral_norm

from EqualizedModules import Conv2d, ConvTranspose2d, Linear, GatedConv2d

class Encoder(nn.Module):
    def __init__(self, init_size=(64, 64),
                        dest_size=(8, 8),
                        scale=0.5,
                        scale_channels=4,
                        output_channels=128,
                        kernel_size=2):
        super(Encoder, self).__init__()

        assert init_size[0] == init_size[1] 
        assert dest_size[0] == dest_size[1]

        self.init_size = init_size
        self.dest_size = dest_size
        self.scale = scale
        self.scale_channels = scale_channels
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        
        self.layer_1 = GatedConv2d(4, 8, kernel_size=4, stride=2, padding=1, bias=False)
        #self.normalization_1 = nn.InstanceNorm2d(8)
        self.layer_2 = GatedConv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False)
        #self.normalization_2 = nn.InstanceNorm2d(16)
        self.layer_3 = GatedConv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False)
        #self.normalization_3 = nn.InstanceNorm2d(32)
        self.layer_4 = GatedConv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False)
        #self.normalization_4 = nn.InstanceNorm2d(64)
        self.layer_5 = GatedConv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        #self.normalization_5 = nn.InstanceNorm2d(128)

        #self.mean = nn.Conv2d(128, 128, kernel_size=2, stride=2)
        self.mean = Linear(128 * 4, 256)
        self.logvar = Linear(128 * 4, 256)
        #self.logvar = nn.Conv2d(128, 128, kernel_size=2, stride=2)

    def forward(self, input):
        x, mask = input
        #noise = torch.randn(x.size(), device=x.device)
        #x = x * (1-mask) + noise * mask
        x = x * (1-mask)
        x = torch.cat((x, mask), dim=1)

        x_1 = self.layer_1(x)
        #x_1 = self.normalization_1(x_1)
        #x_1 = F.leaky_relu(x_1, 0.2, inplace=True)
        
        x_2 = self.layer_2(x_1)
        #x_2 = self.normalization_2(x_2)
        #x_2 = F.leaky_relu(x_2, 0.2, inplace=True)

        x_3 = self.layer_3(x_2)
        #x_3 = self.normalization_3(x_3)
        #x_3 = F.leaky_relu(x_3, 0.2, inplace=True)

        x_4 = self.layer_4(x_3)
        #x_4 = self.normalization_4(x_4)
        #x_4 = F.leaky_relu(x_4, 0.2, inplace=True)

        x_5 = self.layer_5(x_4)
        #x_5 = self.normalization_5(x_5)
        #x_5 = F.leaky_relu(x_5, 0.2)

        #x = x.view(x.size(0), -1)
        #mean = self.dense_mean(x)
        #logvar = self.dense_logvar(x)
        x_5 = x_5.view(x_5.size(0), -1)
        mean = self.mean(x_5)
        logvar = self.logvar(x_5)
        #mean = mean.view(mean.size(0), -1)
        
        #logvar = self.logvar(x)
        #logvar = logvar.view(logvar.size(0), -1)

        return [x_4, x_3, x_2, x_1], mean, logvar

class Generator(nn.Module):

    def __init__(self, init_size=(8, 8),
                        dest_size=(64, 64),
                        scale=1.5,
                        input_channels=128,
                        kernel_size=3,
                        **kwargs):
        super(Generator, self).__init__()

        self.encoder = Encoder()

        self.layer_1 = DenormResBlock(80, 100)
        self.layer_2 = DenormResBlock(132, 64)
        self.layer_3 = DenormResBlock(80, 64)
        self.layer_4 = DenormResBlock(72, 64)
        #self.layer_5 = DenormResBlock(32, 3)

        self.conv = Conv2d(64, 3, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(3)

        self.dense_1 = Linear(256, 256)
        self.dense_2 = Linear(256, 256)
        self.dense_3 = Linear(256, 256)

        self.normalization_1 = nn.BatchNorm1d(256)
        self.normalization_2 = nn.BatchNorm1d(256)
        self.normalization_3 = nn.BatchNorm1d(256)

    def forward(self, input):
        
        z, real, mask = input
        #mean, logvar = self.encoder((real, mask))
        feats, mean, logvar = self.encoder((real, mask))
        #z_1, z_2 = z[:, :128], z[:, 128:]
        #z_1 = z_1 * logvar.mul(0.5).exp() + mean
        #x = F.pixel_shuffle(z_1.view(z_1.size(0), -1, 1, 1), 2)
        
        #print(z.size())
        #z = z.view(-1, z.size(2))
        z = self.dense_1(z)
        z = self.normalization_1(z)
        z = F.leaky_relu(z, 0.2, inplace=True)

        z = self.dense_2(z)
        z = self.normalization_2(z)
        z = F.leaky_relu(z, 0.2, inplace=True)

        z = self.dense_3(z)
        z = self.normalization_3(z)
        z = F.leaky_relu(z, 0.2, inplace=True)
        #print(z.size())
        #z = z.view(-1, 5, z.size(1))

        mean = z * logvar.mul(0.5).exp() + mean
        x = F.pixel_shuffle(mean.view(mean.size(0), -1, 1, 1), 4)
        x = F.leaky_relu(x, 0.2, inplace=True)
        
        #z = self.dense_1(z_2)
        #z = F.leaky_relu(z, 0.2, inplace=True)

        x = torch.cat((x, feats[0]), dim=1)
        x = self.layer_1((z, x, mask))
        x = F.leaky_relu(x, 0.2, inplace=True)
        #x = F.relu(x, inplace=True)

        #z = self.dense_2(z)
        #z = F.leaky_relu(z, 0.2, inplace=True)

        x = torch.cat((x, feats[1]), dim=1)
        x = self.layer_2((z, x, mask))
        x = F.leaky_relu(x, 0.2, inplace=True)
        #x = F.relu(x, inplace=True)


        x = torch.cat((x, feats[2]), dim=1)

        #x = F.relu(x, inplace=True)
        x = self.layer_3((z, x, mask))
        x = F.leaky_relu(x, 0.2, inplace=True)
        #x = F.relu(x, inplace=True)

        #z = F.leaky_relu(z, 0.2, inplace=True)
        #z = self.dense_3(z)
        #z = F.leaky_relu(z, 0.2, inplace=True)
        #x = F.relu(x, inplace=True)
        x = torch.cat((x, feats[3]), dim=1)
        x = self.layer_4((z, x, mask))
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.conv(x)
        x = self.norm(x)

        #x = F.relu(x, inplace=True)

        #x = torch.cat((x, feats[3]), dim=1)
        #x = self.layer_5((z, x, mask))
        x = torch.tanh(x)

        #x = mask * x + (1 - mask) * real

        #x = z * logvar.mul(0.5).exp() + mean
        
        return x, mean, logvar, z
        #return x

class Discriminator(nn.Module):

    def __init__(self, n_feats, is_wgan, noise_params):
        super(Discriminator, self).__init__()
        self.n_feats = n_feats
        self.is_wgan = is_wgan
        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            spectral_norm(Conv2d(4, self.n_feats, 4, stride=2, bias=False, padding=1)),
            #nn.InstanceNorm2d(self.n_feats),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            spectral_norm(Conv2d(self.n_feats, self.n_feats * 2, 4, stride=2, padding=1, bias=False)),
            nn.InstanceNorm2d(self.n_feats * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            spectral_norm(Conv2d(self.n_feats * 2, self.n_feats * 4, 4, stride=1, bias=False)),
            nn.InstanceNorm2d(self.n_feats * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            spectral_norm(Conv2d(self.n_feats * 4, self.n_feats * 8, 4, stride=1, bias=False)),
            nn.InstanceNorm2d(self.n_feats * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            spectral_norm(Conv2d(self.n_feats * 8, self.n_feats * 4, 4, stride=1, bias=False)),
            nn.InstanceNorm2d(self.n_feats * 4),
            nn.LeakyReLU(0.2, inplace=True)
            #nn.BatchNorm2d(self.n_feats),
            #nn.LeakyReLU(0.2, inplace=True),
        )

        self.q_net = AuxiliaryNetwork(self.net, **noise_params)
        self.head_local = nn.Sequential(
            Conv2d(self.n_feats * 4, 1, 3, stride=1, bias=False),
            nn.Flatten()
            #Linear(self.n_feats * 25, self.n_feats * 25, bias=False)
        )
        self.head_global = nn.Sequential(
            Conv2d(self.n_feats * 4, self.n_feats * 2, 4, stride=1, bias=False),
            nn.InstanceNorm2d(self.n_feats * 2),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(self.n_feats * 2, 1, 4, stride=1, bias=False),
            nn.Flatten()
        )
        
    def forward(self, input):
        x, mask, boolean = input
        #x, mask = input

        x = torch.cat([x, mask], dim=1)
        if boolean:
            x = self.net(x)
            output_local = self.head_local(x)
            output_global = self.head_global(x)
            output = torch.cat([output_local, output_global], dim=1)
            return output
        else:
            x = self.net(x)
            output_local = self.head_local(x)
            output_global = self.head_global(x)
            output = torch.cat([output_local, output_global], dim=1)
            mean, var, disc = self.q_net(x)
            return mean, var, disc, output

class AuxiliaryNetwork(nn.Module):

    def __init__(self, net,
                noise_dim, 
                cont_dim,
                disc_dim,
                n_disc):
        
        super(AuxiliaryNetwork, self).__init__()
        self.main = nn.Sequential(
            Conv2d(256, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(128, 64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten()
        )

        self.mean = Linear(64, cont_dim)
        self.var = Linear(64, cont_dim)

        self.disc = Linear(64, disc_dim * n_disc)

    def forward(self, x):
        
        x = self.main(x)

        mean = self.mean(x)
        var = torch.exp(self.var(x))
        disc = self.disc(x)

        return mean, var, disc


if __name__ == '__main__':
    
    '''gen = Generator()
    gen.eval()
    x = torch.randn(10, 3, 64, 64)
    random = torch.randn(10, 128, 8, 8)
    masks = torch.randn(10, 1, 64, 64)
    y = gen((random, x, masks))
    summary(gen, ((128, 8, 8), (3, 64, 64), (1, 64, 64)), device='cpu')'''
    
    #z = torch.randn(10, 3, 64, 64)
    #masks = torch.randn(10, 1, 64, 64)
    #z = torch.stack((z, masks), axis=1)

    #dis = Discriminator().cpu()
    #dis.eval()
    #print(dis.net[5])
    #summary(dis, [(3, 64, 64), (1, 64, 64)], device='cpu')
    #print(dis(torch.randn(10, 3, 64, 64), torch.randn(10, 1, 64, 64)).size())

    enc = Encoder().cpu()
    enc.eval()
    summary(enc, (3, 64, 64), device='cpu')
    x = torch.randn(1, 3, 64, 64)
    print(enc(x).size())

    #x = torch.randn(10, 3, 64, 64)
    #z = enc(x)
    #print(z.size())