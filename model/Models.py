import torch
import torch.nn as nn
from torchsummary import summary
from Denormalization import DenormResBlock, DiscriminatorBlock
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import spectral_norm

from EqualizedModules import Conv2d, ConvTranspose2d, Linear

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
        '''
        # compute n_layers size
        #n_layers = (np.log(self.dest_size[0]) - np.log(self.init_size[0])) / np.log(self.scale)
        #self.n_layers = int(n_layers)

        resolutions = [self.init_size[0] * scale]
        #for i in range(self.n_layers + 1):
        #    resolutions.append(resolutions[-1] * self.scale)
        #resolutions[-1] = self.dest_size[-1]
        while True:
            resolutions.append(resolutions[-1] * scale)
            if resolutions[-1] < self.dest_size[0]:
                break
        resolutions[-1] = self.dest_size[0]
        self.resolutions = list(map(lambda x: torch.Size([int(x)]) * 2, resolutions))
        self.n_layers = len(self.resolutions)

        self.channels = [3]
        for i in range(self.n_layers):
            self.channels.append(int(self.channels[-1] * self.scale_channels))

        self.channels[-1] = self.output_channels
        # set layers
        self.names = []
        for i in range(self.n_layers):
            name = 'conv_{}'.format(i+1)
            setattr(self,
                    name,
                    nn.Sequential(
                        nn.Conv2d(self.channels[i], 
                                self.channels[i+1], 
                                self.kernel_size,
                                stride=2),
                        nn.BatchNorm2d(self.channels[i+1]),
                    )
            )
            self.names.append(name)

        print('Encoder:')
        print('Resolutions: ', self.resolutions)
        print('Channels: ', self.channels)'''

        
        self.layer_1 = Conv2d(4, 8, kernel_size=4, stride=2, padding=1)
        self.normalization_1 = nn.InstanceNorm2d(8)
        self.layer_2 = Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        self.normalization_2 = nn.InstanceNorm2d(16)
        self.layer_3 = Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.normalization_3 = nn.InstanceNorm2d(32)
        self.layer_4 = Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.normalization_4 = nn.InstanceNorm2d(64)
        self.layer_5 = Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.normalization_5 = nn.InstanceNorm2d(128)

        #self.mean = nn.Conv2d(128, 128, kernel_size=2, stride=2)
        self.mean = Linear(128 * 4, 256)
        self.logvar = Linear(128 * 4, 256)
        #self.logvar = nn.Conv2d(128, 128, kernel_size=2, stride=2)
        
        '''
        self.layer_1 = DiscriminatorBlock(3, 8, 3, 128)
        self.layer_2 = DiscriminatorBlock(8, 16, 3, 128)
        self.layer_3 = DiscriminatorBlock(16, 32, 3, 128)
        self.layer_4 = DiscriminatorBlock(32, 64, 3, 128)
        self.layer_5 = DiscriminatorBlock(64, 128, 3, 128)

        self.mean = nn.Linear(128 * 4, 128)'''

    def forward(self, input):
        x, mask = input
        '''for name, resolution in zip(self.names, self.resolutions):

            x = getattr(self, name)(x)
            x = F.leaky_relu(x, 0.2, inplace=True)
            #x = F.interpolate(x, size=resolution, mode='bilinear', align_corners=False)'''

        #noise = torch.randn(x.size(), device=x.device)
        #x = x * (1-mask) + noise * mask
        x = x * (1-mask)
        x = torch.cat((x, mask), dim=1)

        x_1 = self.layer_1(x)
        x_1 = self.normalization_1(x_1)
        x_1 = F.leaky_relu(x_1, 0.2, inplace=True)
        
        x_2 = self.layer_2(x_1)
        x_2 = self.normalization_2(x_2)
        x_2 = F.leaky_relu(x_2, 0.2, inplace=True)

        x_3 = self.layer_3(x_2)
        x_3 = self.normalization_3(x_3)
        x_3 = F.leaky_relu(x_3, 0.2, inplace=True)

        x_4 = self.layer_4(x_3)
        x_4 = self.normalization_4(x_4)
        x_4 = F.leaky_relu(x_4, 0.2, inplace=True)

        x_5 = self.layer_5(x_4)
        x_5 = self.normalization_5(x_5)
        x_5 = F.leaky_relu(x_5, 0.2)

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
        '''
        assert init_size[0] == init_size[1] 
        assert dest_size[0] == dest_size[1]

        self.init_size = init_size
        self.dest_size = dest_size
        self.scale = scale
        self.kernel_size = kernel_size
        self.input_channels = input_channels

        self.encoder = Encoder()

        # compute n_layers size
        n_layers = (np.log(self.dest_size[0]) - np.log(self.init_size[0])) / np.log(self.scale)
        self.n_layers = int(n_layers)

        resolutions = [self.init_size[0]]
        for i in range(self.n_layers + 1):
            resolutions.append(resolutions[-1] * self.scale)
        resolutions[-1] = self.dest_size[-1]
        self.resolutions = list(map(lambda x: torch.Size([int(x)]) * 2, resolutions))

        #print(channels_scale)
        #I scale**(n_layers) = O
        self.channels = [self.input_channels]
        for i in range(self.n_layers + 1):
            self.channels.append(int(self.channels[-1] / self.scale))
        self.channels[-1] = 3

        #print(self.channels)

        self.names = []
        for i in range(self.n_layers + 1):
            name = 'resblock_{}'.format(i+1)
            setattr(self,
                    name,
                    DenormResBlock(self.channels[i], 
                                    self.channels[i+1], 
                                    self.resolutions[i+1],
                                    self.kernel_size),
            )
            self.names.append(name)
        '''

        self.encoder = Encoder()
        #self.const = Constant(128, 8, 8)

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

        '''
        print('Generator:')
        print('Resolutions: ', self.resolutions)
        print('Channels: ', self.channels)'''

    def forward(self, input):
        '''
        z, x, mask = input
        noise_style, noise_masks = z[:, :128], z[:, 128:]
        #x = self.const(x)
        mean, logvar = self.encoder((x, mask))

        noise_style = noise_style * logvar.mul(0.5).exp() + mean
        x = F.pixel_shuffle(noise_style.view(noise_style.size(0), -1, 1, 1), 2)

        z = self.dense_1(noise_masks)
        z = F.leaky_relu(z, 0.2, inplace=True)
        z = self.dense_2(z)
        z = F.leaky_relu(z, 0.2, inplace=True)
        z = self.dense_3(z)
        '''
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

        x = mask * x + (1 - mask) * real

        #x = z * logvar.mul(0.5).exp() + mean
        '''
        for name in self.names:

            x = getattr(self, name)((z, x, mask))
            if x.size(1) != 3:
                x = F.leaky_relu(x, 0.2, inplace=True)
                #x = F.relu(x, inplace=True)
            else:
                x = torch.tanh(x)
        '''
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
            spectral_norm(Conv2d(self.n_feats * 4, self.n_feats * 2, 4, stride=1, bias=False)),
            nn.InstanceNorm2d(self.n_feats * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(Conv2d(self.n_feats * 2, 1, 4, stride=1, bias=False)),
            nn.Flatten()
        )
        
        '''

        self.names = ['block_1',
                    'block_2',
                    'block_3',
                    'block_4',
                    'block_5']
        
        self.block_1 = DiscriminatorBlock(3, 32, 3, 128)
        self.block_2 = DiscriminatorBlock(32, 64, 3, 128)
        self.block_3 = DiscriminatorBlock(64, 128, 3, 128)
        self.block_4 = DiscriminatorBlock(128, 256, 3, 128)
        self.block_5 = DiscriminatorBlock(256, 64, 3, 128)

        self.flatten = nn.Flatten()'''
        
    def forward(self, input):
        x, mask, boolean = input
        #x, mask = input
        '''
        input = self.block_1(input)
        input = self.block_2(input)
        input = self.block_3(input)
        input = self.block_4(input)
        x, _ = self.block_5(input)
        
        x = self.flatten(x)
        #x = self.dense(x)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)'''

        '''
        for name, param in self.named_parameters():
            fan_in = param.data.size(1) * param.data[0][0].numel()
            getattr(self, name).weight.data *= torch.sqrt(2.0 / fan_in) '''
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
        #x = torch.sigmoid(x)   

        #return mean, var, output

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