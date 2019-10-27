import torch
import torch.nn as nn
from torchsummary import summary
from Denormalization import SPADE

class Generator(nn.Module):

    def __init__(self, n_latent=100,
                        n_feats=64):
        super(Generator, self).__init__()
        self.n_latent = n_latent
        self.n_feats = n_feats

        self.deconv_1 = nn.ConvTranspose2d(self.n_latent, self.n_feats * 8, 4, 1, 0, bias=False)
        self.spade_1 = SPADE(self.n_feats * 8, 3)

        self.deconv_2 = nn.ConvTranspose2d(self.n_feats * 8, self.n_feats * 4, 4, 2, 1, bias=False)
        self.spade_2 = SPADE(self.n_feats * 4, 3)

        self.deconv_3 = nn.ConvTranspose2d(self.n_feats * 4, self.n_feats * 2, 4, 2, 1, bias=False)
        self.spade_3 = SPADE(self.n_feats * 2, 3)

        self.deconv_4 = nn.ConvTranspose2d(self.n_feats * 2, self.n_feats, 4, 2, 1, bias=False)
        self.spade_4 = SPADE(self.n_feats, 3)

        self.deconv_5 = nn.ConvTranspose2d(self.n_feats, 3, 4, 2, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, z, mask):

        z = self.deconv_1(z)
        z = self.spade_1(z, mask)
        #z = self.leaky_relu(z)
        z = self.relu(z)

        z = self.deconv_2(z)
        z = self.spade_2(z, mask)
        #z = self.leaky_relu(z)
        z = self.relu(z)

        z = self.deconv_3(z)
        z = self.spade_3(z, mask)
        #z = self.leaky_relu(z)
        z = self.relu(z)

        z = self.deconv_4(z)
        z = self.spade_4(z, mask)
        #z = self.leaky_relu(z)
        z = self.relu(z)

        z = self.deconv_5(z)
        z = self.tanh(z)

        return z

class Discriminator(nn.Module):

    def __init__(self, n_feats=64):
        super(Discriminator, self).__init__()
        self.n_feats = n_feats
        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, self.n_feats, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.n_feats, self.n_feats * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_feats * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.n_feats * 2, self.n_feats * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_feats * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.n_feats * 4, self.n_feats * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.n_feats * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.n_feats * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':

    gen = Generator()
    gen.eval()
    x = torch.randn(10, 100, 1, 1)
    masks = torch.randn(10, 1, 64, 64)
    y = gen(x, masks)
    summary(gen, [(100, 1, 1), (1, 64, 64)])
    print(y.size())

    dis = Discriminator()
    dis.eval()
    x = torch.randn(10, 3, 64, 64)
    y = dis(x)
    print(y.size())
    print(y)
