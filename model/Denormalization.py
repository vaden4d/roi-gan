
import torch
import torch.nn as nn
import torch.nn.functional as F

from EqualizedModules import Conv2d, ConvTranspose2d, Linear

class SPADE(nn.Module):

    def __init__(self, n_channels, kernel_size, n_hidden, control=True):
        super().__init__()

        self.control = control
        self.normalization = nn.InstanceNorm2d(n_channels, affine=False)
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.n_channels = n_channels

        self.shared = Conv2d(1, self.n_hidden, kernel_size=self.kernel_size)
        self.gamma = Conv2d(self.n_hidden, n_channels, kernel_size=self.kernel_size)
        self.beta = Conv2d(self.n_hidden, n_channels, kernel_size=self.kernel_size)

        self.dense_transform = Linear(256, 256)

    def forward(self, input):

        z, x, mask = input

        # normalize input
        normalized = self.normalization(x)

        # beta and gamma generation
        size_x = x.size(2) + 2 * self.kernel_size - 2
        size_y = x.size(3) + 2 * self.kernel_size - 2

        mask = F.interpolate(mask, size=(size_x, size_y), mode='nearest')
        
        mask = self.shared(mask)
        mask = F.leaky_relu(mask, 0.2, inplace=True)
        # if control -- control style inside mask
        if self.control:
            z = self.dense_transform(z)
            mask = (1 + z[:, :self.n_hidden].view(z.size(0), self.n_hidden, 1, 1)) * mask + z[:, self.n_hidden:].view(z.size(0), self.n_hidden, 1, 1)

        gamma = self.gamma(mask)
        beta = self.beta(mask)

        gamma = F.leaky_relu(gamma, 0.2, inplace=True)
        beta = F.leaky_relu(beta, 0.2, inplace=True)

        out = normalized * (1 + gamma) + beta 

        return out

# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
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
        self.conv_left = Conv2d(self.input_channels,
                                    self.output_channels,
                                    self.kernel_size,
                                    padding=self.padding)
        '''

        self.deconv_left = ConvTranspose2d(self.input_channels,
                                               self.output_channels,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1)
        '''

        self.spade_left = SPADE(self.input_channels, self.kernel_size, 128, True)
        self.spade_left_2 = SPADE(self.output_channels, self.kernel_size, 128, True)
        # right branch
        self.conv_right = Conv2d(self.input_channels,
                                    self.output_channels,
                                    self.kernel_size,
                                    padding=self.padding)

        '''self.deconv_right = ConvTranspose2d(self.input_channels,
                                               self.output_channels,
                                               kernel_size=4,
                                               stride=2,
                                               padding=1)'''
        
        self.spade_right = SPADE(self.input_channels, self.kernel_size, 128, False)
        self.spade_right_2 = SPADE(self.output_channels, self.kernel_size, 128, False)
        
        self.spade_last = SPADE(self.output_channels, self.kernel_size, 128, True)

        #self.normalization_left = nn.BatchNorm2d(self.output_channels * 4)
        #self.normalization_right = nn.BatchNorm2d(self.output_channels * 4)

        self.conv_last = Conv2d(self.output_channels,
                                    self.output_channels,
                                    self.kernel_size,
                                    padding=self.padding)

    def forward(self, input):

        z, x, mask = input
        
        # left branch
        left = self.spade_left((z, x, mask))
        left = F.leaky_relu(left, 0.2, inplace=True)
        left = self.conv_left(left)
        #left = self.normalization_left(left)
        left = F.leaky_relu(left, 0.2, inplace=True)

        # right branch
        right = self.spade_right((z, x, 1-mask))
        right = F.leaky_relu(right, 0.2, inplace=True)
        right = self.conv_right(x)
        #right = self.normalization_right(right)
        right = F.leaky_relu(right, 0.2, inplace=True)
        
        '''
        left = torch.cat([left, right], dim=1)
        left = F.leaky_relu(left, 0.2, inplace=True)
        
        left = F.pixel_shuffle(left, 2)
        left = self.conv(left)'''

        #left = left + right
        #left = F.pixel_shuffle(left, 2)
        #right = F.pixel_shuffle(right, 2)

        #left = self.deconv_left(left)
        #right = self.deconv_right(right)

        #left = self.spade_left_2((z, left, mask))
        #right = self.spade_right_2((z, right, 1-mask))

        left = F.upsample_nearest(left, scale_factor=2)
        right = F.upsample_nearest(right, scale_factor=2)

        left = left + right

        left = self.conv_last(left)
        left = F.leaky_relu(left, 0.2, inplace=True)

        #left = F.pixel_shuffle(left, 2)

        left = self.spade_last((z, left, mask))

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
