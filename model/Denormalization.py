
import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):

    def __init__(self, n_channels, kernel_size, n_hidden):
        super().__init__()

        self.normalization = nn.BatchNorm2d(n_channels, affine=False)
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.n_channels = n_channels

        # hidden layer
        #n_hidden = n_channels // 2
        '''n_hidden = 128
        self.shared = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=self.kernel_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mask_normalization = nn.BatchNorm2d(n_hidden, affine=False)

        self.conv_gamma = nn.Conv2d(n_hidden, n_channels, kernel_size=self.kernel_size)
        self.conv_beta = nn.Conv2d(n_hidden, n_channels, kernel_size=self.kernel_size)

        self.dense = nn.Sequential(
            nn.Linear(128, 2 * 128),
            nn.LeakyReLU(0.2, inplace=True)
        )'''

        self.mask_conv = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=self.kernel_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.non_mask_conv = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=self.kernel_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        #self.dense = nn.Sequential(
        #    nn.Linear(128, 2 * 128),
        #    nn.LeakyReLU(0.2, inplace=True)
        #)

        self.gamma = nn.Conv2d(n_hidden, n_channels, kernel_size=self.kernel_size)
        self.beta = nn.Conv2d(n_hidden, n_channels, kernel_size=self.kernel_size)

    def forward(self, input):

        z, x, mask = input

        # normalize input
        normalized = self.normalization(x)

        # beta and gamma generation
        size_x = x.size(2) + 2 * self.kernel_size - 2
        size_y = x.size(3) + 2 * self.kernel_size - 2

        mask = F.interpolate(mask, size=(size_x, size_y), mode='nearest')

        gamma = self.non_mask_conv(1 - mask)
        beta = self.mask_conv(mask)

        #z = self.dense(z)
        #print(z.size())
        #print(beta.size())
        beta = (1 + z[:, :self.n_hidden].view(z.size(0), self.n_hidden, 1, 1)) * beta + z[:, self.n_hidden:].view(z.size(0), self.n_hidden, 1, 1)

        gamma = self.gamma(gamma)
        beta = self.beta(beta)

        out = normalized * (1 + gamma) + beta
        '''
        
        mask = F.interpolate(mask, size=(size_x, size_y), mode='nearest')
        
        activations = self.shared(mask)
        activations = self.mask_normalization(activations)

        z = self.dense(z)
        activations = (1 + z[:, :128].view(z.size(0), 128, 1, 1)) * activations + z[:, 128:].view(z.size(0), 128, 1, 1)
        
        gamma = self.conv_gamma(activations)
        beta = self.conv_beta(activations)

        #z = self.dense(z)

        #gamma = z[:, :self.n_channels].view(z.size(0), self.n_channels, 1, 1) * gamma + z[:, self.n_channels:].view(z.size(0), self.n_channels, 1, 1)
        #beta = z[:, :self.n_channels].view(z.size(0), self.n_channels, 1, 1) * beta + z[:, self.n_channels:].view(z.size(0), self.n_channels, 1, 1)

        mask = F.interpolate(mask, size=gamma.size()[2:], mode='nearest')
        # apply scale and bias
        out = normalized * (1 + gamma) + beta 
        #out = normalized * (1 + gamma) + beta + noise * self.weight.view(1, -1, 1, 1) '''

        return out


class DenormResBlock(nn.Module):

    def __init__(self, input_channels, 
                        output_channels, 
                        output_size,
                        kernel_size=3):
        super(DenormResBlock, self).__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels

        # left branch
        self.n_hidden = 128
        self.spade_1 = SPADE(input_channels, self.kernel_size, self.n_hidden)
        self.deconv_1 = nn.Conv2d(self.input_channels, 
                                            self.output_channels,
                                            self.kernel_size,
                                            stride=1,
                                            bias=False)
        self.spade_2 = SPADE(output_channels, self.kernel_size, self.n_hidden)
        self.deconv_2 = nn.Conv2d(self.output_channels, 
                                            self.output_channels,
                                            self.kernel_size,
                                            stride=1,
                                            bias=False)

        # right branch
        self.spade_3 = SPADE(input_channels, self.kernel_size, self.n_hidden)
        self.deconv_3 = nn.Conv2d(self.input_channels, 
                                            self.output_channels,
                                            2 * self.kernel_size - 1,
                                            stride=1,
                                            bias=False)

        self.dense_1 = nn.Linear(256, 256)

    def forward(self, input):

        z, x, mask = input

        # left branch
        left = self.spade_1((z, x, mask))
        left = F.relu(left, inplace=True)
        left = self.deconv_1(left)
        
        left = self.spade_2((z, left, mask))
        left = F.relu(left, inplace=True)
        left = self.deconv_2(left)

        # right branch
        right = self.spade_3((z, x, mask))
        right = F.relu(right, inplace=True)
        right = self.deconv_3(right)
        left = left + right

        left = F.interpolate(left, size=self.output_size, mode='bilinear', align_corners=False)

        return left

if __name__ == '__main__':
    '''
    x = torch.randn(10, 5, 30, 30)
    mask = torch.randn(10, 1, 13, 13)
    obj = SPADE(5, 3, 128)
    z = obj((x, mask))
    print(z.size())'''

    x = torch.randn(100, 128, 1, 1)
    mask = torch.randn(100, 1, 64, 64)
    obj = DenormResBlock(128, 100, 2)
    z = obj((x, mask))
    print(z.size())
