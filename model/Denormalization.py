
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorBlock(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, n_hidden):
        super().__init__()

        self.normalization = nn.BatchNorm2d(n_channels, affine=False)
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        
        self.input_channels = input_channels
        self.output_channels = output_channels

        # hidden layer
        self.shared = nn.Conv2d(1, self.n_hidden, kernel_size=self.kernel_size)

        self.gamma = nn.Conv2d(self.n_hidden, n_channels, kernel_size=self.kernel_size)
        self.beta = nn.Conv2d(self.n_hidden, n_channels, kernel_size=self.kernel_size)

    def forward(self, input):

        z, x, mask = input

        # normalize input
        normalized = self.normalization(x)

        # beta and gamma generation
        size_x = x.size(2) + 2 * self.kernel_size - 2
        size_y = x.size(3) + 2 * self.kernel_size - 2

        mask_ = F.interpolate(mask, size=(size_x, size_y), mode='nearest')
        
        mask_ = self.shared(mask_)
        mask_ = F.leaky_relu(mask_, 0.2, inplace=True)

        gamma = self.gamma(mask_)
        beta = self.beta(mask_)

        out = normalized * (1 + gamma) + beta

        return out, mask

class SPADE(nn.Module):

    def __init__(self, n_channels, kernel_size, n_hidden):
        super().__init__()

        self.normalization = nn.BatchNorm2d(n_channels, affine=False)
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.n_channels = n_channels

        # hidden layer
        self.shared = nn.Conv2d(1, self.n_hidden, kernel_size=self.kernel_size)

        self.gamma = nn.Conv2d(self.n_hidden, n_channels, kernel_size=self.kernel_size)
        self.beta = nn.Conv2d(self.n_hidden, n_channels, kernel_size=self.kernel_size)

    def forward(self, input):

        z, x, mask = input

        # normalize input
        normalized = self.normalization(x)

        # beta and gamma generation
        size_x = x.size(2) + 2 * self.kernel_size - 2
        size_y = x.size(3) + 2 * self.kernel_size - 2

        mask = F.interpolate(mask, size=(size_x, size_y), mode='nearest')
        
        mask = self.shared(mask)
        mask = (1 + z[:, :self.n_hidden].view(z.size(0), self.n_hidden, 1, 1)) * mask + z[:, self.n_hidden:].view(z.size(0), self.n_hidden, 1, 1)
        mask = F.leaky_relu(mask, 0.2, inplace=True)

        gamma = self.gamma(mask)
        beta = self.beta(mask)

        #mask = F.interpolate(mask, size=gamma.size()[2:], mode='nearest')

        out = normalized * (1 + gamma) + beta

        return out

class DenormResBlock(nn.Module):

    def __init__(self, input_channels, 
                        output_channels, 
                        kernel_size=3):
        super(DenormResBlock, self).__init__()
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.padding = (self.kernel_size - 1) // 2
        
        # left branch
        self.conv_left = nn.Conv2d(self.input_channels,
                                    self.input_channels * 2,
                                    self.kernel_size,
                                    padding=self.padding)

        self.spade_left = SPADE(self.input_channels, self.kernel_size, 128)

        # right branch
        self.conv_right = nn.Conv2d(self.input_channels,
                                    self.input_channels * 2,
                                    self.kernel_size,
                                    padding=self.padding)
        
        self.spade_right = SPADE(self.input_channels, self.kernel_size, 128)

        # the last layer
        self.conv = nn.Conv2d(self.input_channels,
                                self.output_channels,
                                self.kernel_size,
                                padding=self.padding)


    def forward(self, input):

        z, x, mask = input
        
        # left branch
        left = self.spade_left((z, x, mask))
        left = self.conv_left(left)
        left = F.leaky_relu(left, 0.2, inplace=True)

        # right branch
        right = self.spade_right((z, x, mask))
        right = self.conv_right(right)
        right = F.leaky_relu(right, 0.2, inplace=True)
        
        left = torch.cat([left, right], dim=1)
        left = F.leaky_relu(left, 0.2, inplace=True)
        
        left = F.pixel_shuffle(left, 2)
        left = self.conv(left)

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
