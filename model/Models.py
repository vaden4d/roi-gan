import torch
import torch.nn as nn
from torchsummary import summary
from Denormalization import DenormResBlock
import torch.nn.functional as F
import numpy as np

from torch.nn.utils import spectral_norm

class Constant(nn.Module):

    def __init__(self, *tensor_size):
        super(Constant, self).__init__()

        self.constant = nn.Parameter(torch.randn(1, *tensor_size))

    def forward(self, x):
        constant = self.constant.repeat(x.size(0), 1, 1, 1)
        return constant

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

        self.layer_1 = nn.Conv2d(3, 8, kernel_size=2, stride=2)
        self.layer_2 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
        self.layer_3 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.layer_4 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.layer_5 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.mean = nn.Conv2d(128, 128, kernel_size=2, stride=2)
        self.logvar = nn.Conv2d(128, 128, kernel_size=2, stride=2)


    def forward(self, x):
        
        '''for name, resolution in zip(self.names, self.resolutions):

            x = getattr(self, name)(x)
            x = F.leaky_relu(x, 0.2, inplace=True)
            #x = F.interpolate(x, size=resolution, mode='bilinear', align_corners=False)'''
        
        x = self.layer_1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.layer_2(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.layer_3(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.layer_4(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.layer_5(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        #x = self.layer_6(x)
        #x = F.leaky_relu(x, 0.2)

        #x = x.view(x.size(0), -1)
        #mean = self.dense_mean(x)
        #logvar = self.dense_logvar(x)
        mean = self.mean(x)
        mean = mean.view(mean.size(0), -1)
        logvar = self.logvar(x)
        logvar = logvar.view(logvar.size(0), -1)

        return mean, logvar

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

        self.layer_1 = DenormResBlock(32, 64)
        self.layer_2 = DenormResBlock(64, 32)
        self.layer_3 = DenormResBlock(32, 16)
        self.layer_4 = DenormResBlock(16, 8)
        self.layer_5 = DenormResBlock(8, 3)

        self.dense_1 = nn.Linear(128, 256)
        self.dense_2 = nn.Linear(256, 256)
        self.dense_3 = nn.Linear(256, 256)
        '''
        print('Generator:')
        print('Resolutions: ', self.resolutions)
        print('Channels: ', self.channels)'''

    def forward(self, input):

        z, x, mask = input
        #x = self.const(x)
        mean, logvar = self.encoder(x)

        z = z * logvar.mul(0.5).exp() + mean
        x = F.pixel_shuffle(z.view(z.size(0), -1, 1, 1), 2)

        z = self.dense_1(z)
        z = F.leaky_relu(z, 0.2, inplace=True)
        z = self.dense_2(z)
        z = F.leaky_relu(z, 0.2, inplace=True)
        z = self.dense_3(z)
        
        x = self.layer_1((z, x, mask))
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.layer_2((z, x, mask))
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.layer_3((z, x, mask))
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.layer_4((z, x, mask))
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.layer_5((z, x, mask))

        #x = torch.tanh(x)

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
        return mean, logvar, x
        #return x

class Discriminator(nn.Module):

    def __init__(self, n_feats=128, scale=1.2, is_wgan=False):
        super(Discriminator, self).__init__()
        self.n_feats = n_feats
        self.is_wgan = is_wgan
        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(4, self.n_feats, 2, 2, bias=False),
            nn.BatchNorm2d(self.n_feats),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.n_feats, self.n_feats * 2, 2, bias=False),
            nn.BatchNorm2d(self.n_feats * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.n_feats * 2, self.n_feats * 4, 2, 2, bias=False),
            nn.BatchNorm2d(self.n_feats * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.n_feats * 4, self.n_feats * 4, 2, 2, bias=False),
            nn.BatchNorm2d(self.n_feats * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.n_feats * 4, self.n_feats * 4, 2, 2, bias=False),
            nn.BatchNorm2d(self.n_feats * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.n_feats * 4, self.n_feats * 4, 2, 2, bias=False),
            nn.BatchNorm2d(self.n_feats * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.dense = nn.Linear(512, 1, bias=False)
        
    def forward(self, input):
        x, mask = input
        # set to empty list with intermidiate
        # layers
        #self.int_outputs = []
        # feature extraction
        x = torch.cat([x, mask], dim=1)
        x = self.net(x)
        #for module in self.net:
        #
        #    x = module((x, mask))
        #    x = F.leaky_relu(x, 0.2, inplace=True)

        #x = self.net((x, mask))
        #x = self.pooling(x)
        # sigmoid
        
        #x = x.view(-1, 1)
        #if ~self.is_wgan:
        #    x = torch.sigmoid(x)
        #    x = x.mean(axis=[1, 2, 3])
        #else:
        #    x = x.mean(axis=[1, 2, 3])
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        #x = torch.sigmoid(x)

        return x

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