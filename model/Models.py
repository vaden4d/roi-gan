import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchsummary import summary
from Denormalization import DenormResBlock
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, n_output=128):
        super(Encoder, self).__init__()

        self.layer_1 = nn.Conv2d(3, 8, kernel_size=2, stride=2)
        self.layer_2 = nn.Conv2d(8, 16, kernel_size=2, stride=2)
        self.layer_3 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.layer_4 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.layer_5 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        self.conv_mean = nn.Conv2d(128, n_output, kernel_size=2, stride=2)
        self.conv_logvar = nn.Conv2d(128, n_output, kernel_size=2, stride=2)


    def forward(self, x):

        x = self.layer_1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.layer_2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.layer_3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.layer_4(x)
        x = F.leaky_relu(x, 0.2)
        x = self.layer_5(x)
        x = F.leaky_relu(x, 0.2)

        #x = x.view(x.size(0), -1)
        #mean = self.dense_mean(x)
        #logvar = self.dense_logvar(x)
        mean = self.conv_mean(x)
        logvar = self.conv_logvar(x)

        return mean, logvar

class Generator(nn.Module):

    def __init__(self, n_input=4,
                        n_spade=128,
                        kernel_size=2):
        super(Generator, self).__init__()
        self.n_input = n_input
        self.n_spade = n_spade
        self.kernel_size = kernel_size

        self.encoder = Encoder()

        self.resblock_1 = DenormResBlock(128, 100, self.kernel_size)
        self.resblock_2 = DenormResBlock(100, 64, self.kernel_size)
        self.resblock_3 = DenormResBlock(64, 32, self.kernel_size)
        self.resblock_4 = DenormResBlock(32, 16, self.kernel_size)
        self.resblock_5 = DenormResBlock(16, 8, self.kernel_size)
        self.resblock_6 = DenormResBlock(8, 3, self.kernel_size)


    def forward(self, z, x, mask):

        mean, logvar = self.encoder(x)
        x = z * logvar.mul(0.5).exp() + mean

        x = self.resblock_1(x, mask)
        x = self.resblock_2(x, mask)
        x = self.resblock_3(x, mask)
        x = self.resblock_4(x, mask)
        x = self.resblock_5(x, mask)
        x = self.resblock_6(x, mask)
        x = torch.tanh(x)

        return mean, logvar, x

class Discriminator(nn.Module):

    def __init__(self, n_feats=128):
        super(Discriminator, self).__init__()
        self.n_feats = n_feats
        self.int_outputs = []
        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(4, self.n_feats // 4, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.n_feats // 4, self.n_feats // 2, 2, bias=False),
            nn.BatchNorm2d(self.n_feats // 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.n_feats // 2, self.n_feats, 2, 2, bias=False),
            nn.BatchNorm2d(self.n_feats),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.n_feats, self.n_feats * 2, 2, 2, bias=False),
            nn.BatchNorm2d(self.n_feats * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.n_feats * 2, self.n_feats, 2, 2, bias=False),
            nn.BatchNorm2d(self.n_feats),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.n_feats, 1, 2, 2, bias=False)
        )
        
    def forward(self, x, mask):
        # set to empty list with intermidiate
        # layers
        self.int_outputs = []

        # feature extraction
        x = torch.cat([x, mask], dim=1)
        x = self.net(x)

        # sigmoid
        x = x.view(-1, 1)
        x = torch.sigmoid(x)

        return x

if __name__ == '__main__':
    
    gen = Generator()
    gen.eval()
    x = torch.randn(10, 128, 1, 1)
    masks = torch.randn(10, 1, 64, 64)
    #y = gen(x, masks)
    #print(y.size())
    summary(gen, [(128, 1, 1), (1, 64, 64)], device='cpu')
    
    #z = torch.randn(10, 3, 64, 64)
    #masks = torch.randn(10, 1, 64, 64)
    #z = torch.stack((z, masks), axis=1)

    #dis = Discriminator().cpu()
    #dis.eval()
    #print(dis.net[5])
    #summary(dis, [(3, 64, 64), (1, 64, 64)], device='cpu')
    #print(dis(torch.randn(10, 3, 64, 64), torch.randn(10, 1, 64, 64)).size())

    #enc = Encoder().cpu()
    #enc.eval()
    #print(enc.layer_1)

    #x = torch.randn(10, 3, 64, 64)
    #z = enc(x)
    #print(z.size())