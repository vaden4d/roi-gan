
import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):

    def __init__(self, n_channels, kernel_size):
        super().__init__()

        self.normalization = nn.BatchNorm2d(n_channels, affine=False)

        # hidden layer
        #n_hidden = n_channels // 2
        n_hidden = 128
        self.shared = nn.Sequential(
            nn.ConvTranspose2d(1, n_hidden, kernel_size=kernel_size),
            nn.ReLU()
        )

        self.conv_gamma = nn.Conv2d(n_hidden, n_channels, kernel_size=kernel_size)
        self.conv_beta = nn.Conv2d(n_hidden, n_channels, kernel_size=kernel_size)

    def forward(self, x, mask):

        # normalize input
        normalized = self.normalization(x)

        # beta and gamma generation
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')

        activations = self.shared(mask)
        gamma = self.conv_gamma(activations)
        beta = self.conv_beta(activations)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class DenormResBlock(nn.Module):

    def __init__(self, input_channels, 
                        output_channels, 
                        kernel_size=3,
                        n_spade=128):
        super(DenormResBlock, self).__init__()
        self.n_spade = n_spade
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels

        # left branch
        self.spade_1 = SPADE(input_channels, self.kernel_size)
        self.deconv_1 = nn.ConvTranspose2d(self.input_channels, 
                                            self.output_channels,
                                            self.kernel_size,
                                            stride=1,
                                            bias=False)
        self.spade_2 = SPADE(output_channels, self.kernel_size)
        self.deconv_2 = nn.ConvTranspose2d(self.output_channels, 
                                            self.output_channels,
                                            self.kernel_size,
                                            stride=1,
                                            bias=False)

        # right branch
        self.spade_3 = SPADE(input_channels, self.kernel_size)
        self.deconv_3 = nn.ConvTranspose2d(self.input_channels, 
                                            self.output_channels,
                                            self.kernel_size,
                                            stride=2,
                                            bias=False)

                                            # (H0-1)2 + k
                                            # H0 + H0 - k - 2
                                            # H0 + 2 (H0 // 2 - k // 2 - 1)
                                            # H0 + 2 ( (H0 - k) // 2 - 1 ) 

                                            # H0 + 2(k - 1) 

    def forward(self, x, mask):

        # left branch
        left = self.spade_1(x, mask)
        left = F.relu(left)
        left = self.deconv_1(left)
        
        left = self.spade_2(left, mask)
        left = F.relu(left)
        left = self.deconv_2(left)

        # right branch
        right = self.spade_3(x, mask)
        right = F.relu(right)
        right = self.deconv_3(right)

        left = F.interpolate(left, size=right.size()[2:], mode='bilinear', align_corners=False)

        return left + right

if __name__ == '__main__':
    '''
    x = torch.randn(10, 5, 30, 30)
    mask = torch.randn(10, 1, 13, 13)
    obj = SPADE(5, 3, 128)
    z = obj(x, mask)
    print(z.size())'''

    x = torch.randn(100, 128, 1, 1)
    mask = torch.randn(100, 1, 64, 64)
    obj = DenormResBlock(128, 100, 2)
    z = obj(x, mask)
    print(z.size())
