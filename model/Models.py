import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchsummary import summary
from Denormalization import DenormResBlock
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, n_input=4,
                        n_spade=128,
                        kernel_size=7):
        super(Generator, self).__init__()
        self.n_input = n_input
        self.n_spade = n_spade
        self.kernel_size = kernel_size
        '''
        self.deconv_1 = spectral_norm(nn.ConvTranspose2d(self.n_latent, self.n_feats * 8, 4, 1, 0, bias=False))
        self.spade_1 = SPADE(self.n_feats * 8, 3, n_hidden)

        self.deconv_2 = spectral_norm(nn.ConvTranspose2d(self.n_feats * 8, self.n_feats * 4, 4, 2, 1, bias=False))
        self.spade_2 = SPADE(self.n_feats * 4, 3, n_hidden)

        self.deconv_3 = spectral_norm(nn.ConvTranspose2d(self.n_feats * 4, self.n_feats * 2, 4, 2, 1, bias=False))
        self.spade_3 = SPADE(self.n_feats * 2, 3, n_hidden)

        self.deconv_4 = spectral_norm(nn.ConvTranspose2d(self.n_feats * 2, self.n_feats, 4, 2, 1, bias=False))
        self.spade_4 = SPADE(self.n_feats, 3, n_hidden)

        self.deconv_5 = spectral_norm(nn.ConvTranspose2d(self.n_feats, 3, 4, 2, 1, bias=False))

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()'''

        self.resblock_1 = DenormResBlock(4, 8, self.kernel_size)
        self.resblock_2 = DenormResBlock(8, 16, self.kernel_size)
        self.resblock_3 = DenormResBlock(16, 32, self.kernel_size)
        self.resblock_4 = DenormResBlock(32, 16, self.kernel_size)
        self.resblock_5 = DenormResBlock(16, 3, self.kernel_size)

    def forward(self, x, mask):
        '''
        z = self.deconv_1(z)
        z = self.spade_1(z, mask)
        z = self.relu(z)

        z = self.deconv_2(z)
        z = self.spade_2(z, mask)
        z = self.relu(z)

        z = self.deconv_3(z)
        z = self.spade_3(z, mask)
        z = self.relu(z)

        z = self.deconv_4(z)
        z = self.spade_4(z, mask)
        z = self.relu(z)

        z = self.deconv_5(z)
        z = self.tanh(z)'''

        x = self.resblock_1(x, mask)
        x = self.resblock_2(x, mask)
        x = self.resblock_3(x, mask)
        x = self.resblock_4(x, mask)
        x = self.resblock_5(x, mask)
        x = torch.tanh(x)

        return x

class Discriminator(nn.Module):

    def __init__(self, n_feats=128):
        super(Discriminator, self).__init__()
        self.n_feats = n_feats
        self.int_outputs = []
        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            spectral_norm(nn.Conv2d(4, self.n_feats // 4, 2, 2, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            spectral_norm(nn.Conv2d(self.n_feats // 4, self.n_feats // 2, 2, bias=False)),
            nn.InstanceNorm2d(self.n_feats // 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            spectral_norm(nn.Conv2d(self.n_feats // 2, self.n_feats, 2, 2, bias=False)),
            nn.InstanceNorm2d(self.n_feats),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            spectral_norm(nn.Conv2d(self.n_feats, self.n_feats * 2, 2, 2, bias=False)),
            nn.InstanceNorm2d(self.n_feats * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            spectral_norm(nn.Conv2d(self.n_feats * 2, self.n_feats * 4, 2, 2, bias=False)),
            nn.InstanceNorm2d(self.n_feats * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            spectral_norm(nn.Conv2d(self.n_feats * 4, self.n_feats * 2, 2, 2, bias=False))
        )
        
    def forward(self, x, mask):
        # set to empty list with intermidiate
        # layers
        self.int_outputs = []

        # feature extraction
        x = torch.cat([x, mask], dim=1)
        x = self.net(x)

        # sigmoid
        x = x.mean(axis=1)
        x = x.view(-1, 1)
        x = torch.sigmoid(x)

        return x

if __name__ == '__main__':
    
    gen = Generator()
    gen.eval()
    x = torch.randn(10, 4, 4, 4)
    masks = torch.randn(10, 1, 64, 64)
    y = gen(x, masks)
    #print(y.size())
    #summary(gen, [(4, 4, 4), (1, 64, 64)], device='cpu')
    
    #z = torch.randn(10, 3, 64, 64)
    #masks = torch.randn(10, 1, 64, 64)
    #z = torch.stack((z, masks), axis=1)

    dis = Discriminator().cpu()
    dis.eval()

    summary(dis, (4, 64, 64), device='cpu')
    print(dis(torch.randn(10, 4, 64, 64)).size())