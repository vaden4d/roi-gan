
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorBlock(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, n_hidden):
        super().__init__()
        '''
        self.normalization = nn.BatchNorm2d(input_channels, affine=False)
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        
        self.input_channels = input_channels
        self.output_channels = output_channels

        # hidden layer
        self.shared_mask = nn.Conv2d(1, self.n_hidden, kernel_size=self.kernel_size)
        self.shared_non_mask = nn.Conv2d(1, self.n_hidden, kernel_size=self.kernel_size)

        self.gamma = nn.Conv2d(self.n_hidden, self.input_channels, kernel_size=self.kernel_size)
        self.beta = nn.Conv2d(self.n_hidden, self.input_channels, kernel_size=self.kernel_size)

        # the last conv layer
        self.conv = nn.Conv2d(self.input_channels, self.output_channels, 
                                kernel_size=2, stride=2)
        '''

        self.normalization = nn.BatchNorm2d(input_channels, affine=False)
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        self.shared = nn.Conv2d(1, self.n_hidden, kernel_size=self.kernel_size)
        self.gamma = nn.Conv2d(self.n_hidden, self.input_channels, kernel_size=self.kernel_size)
        self.beta = nn.Conv2d(self.n_hidden, self.input_channels, kernel_size=self.kernel_size)

        # the last conv layer
        self.conv = nn.Conv2d(self.input_channels, self.output_channels, 
                                kernel_size=2, stride=2)


    def forward(self, input):
        
        x, mask = input
        
        # normalize input
        normalized = self.normalization(x)

        # beta and gamma generation
        size_x = x.size(2) + 2 * self.kernel_size - 2
        size_y = x.size(3) + 2 * self.kernel_size - 2

        mask_ = F.interpolate(mask, size=(size_x, size_y), mode='nearest')

        mask_ = self.shared(mask_)
        mask_ = F.relu(mask_, inplace=True)
        gamma = self.gamma(mask_)
        beta = self.beta(mask_)

        out = normalized * (1 + gamma) + beta
        out = self.conv(out)
        out = F.leaky_relu(out, 0.2, inplace=True)
        '''
        mask_zone = self.shared_mask(mask_)
        mask_zone = F.relu(mask_zone, inplace=True)

        non_mask = self.shared_non_mask(1 - mask_)
        non_mask = F.relu(non_mask, inplace=True)

        gamma = self.gamma(mask_zone)
        beta = self.beta(non_mask)

        out = normalized * (1 + gamma) + x * beta 

        out = self.conv(out)
        out = F.leaky_relu(out, 0.2, inplace=True)'''

        return out, mask

class SPADE(nn.Module):

    def __init__(self, n_channels, kernel_size, n_hidden, control=True):
        super().__init__()

        '''
        self.normalization = nn.BatchNorm2d(n_channels, affine=False)
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.n_channels = n_channels

        # hidden layer
        self.shared = nn.Conv2d(1, self.n_hidden, kernel_size=self.kernel_size)
        self.shared_non_mask = nn.Conv2d(1, self.n_hidden, kernel_size=self.kernel_size)

        self.gamma = nn.Conv2d(self.n_hidden, n_channels, kernel_size=self.kernel_size)
        self.beta = nn.Conv2d(self.n_hidden, n_channels, kernel_size=self.kernel_size)'''

        self.control = control
        self.normalization = nn.BatchNorm2d(n_channels, affine=False)
        self.mask_normalization = nn.InstanceNorm2d(n_hidden, affine=False)
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.n_channels = n_channels

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
        '''
        mask_ = self.shared(mask)
        mask_ = F.relu(mask_, inplace=True)

        non_mask_ = self.shared_non_mask(1 - mask)
        non_mask_ = F.relu(non_mask_, inplace=True)

        mask_ = (1 + z[:, :self.n_hidden].view(z.size(0), self.n_hidden, 1, 1)) * mask_ + z[:, self.n_hidden:].view(z.size(0), self.n_hidden, 1, 1)

        
        '''
        mask = self.shared(mask)
        mask = self.mask_normalization(mask)
        mask = F.leaky_relu(mask, 0.2, inplace=True)
        if self.control:
            mask = (1 + z[:, :self.n_hidden].view(z.size(0), self.n_hidden, 1, 1)) * mask + z[:, self.n_hidden:].view(z.size(0), self.n_hidden, 1, 1)
        #mask = F.leaky_relu(mask, 0.2, inplace=True)

        gamma = self.gamma(mask)
        beta = self.beta(mask)

        #mask = F.interpolate(mask, size=gamma.size()[2:], mode='nearest')

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
        self.conv_left = nn.Conv2d(self.input_channels,
                                    self.output_channels * 4,
                                    self.kernel_size,
                                    padding=self.padding)

        self.spade_left = SPADE(self.input_channels, self.kernel_size, 128, True)

        # right branch
        self.conv_right = nn.Conv2d(self.input_channels,
                                    self.output_channels * 4,
                                    self.kernel_size,
                                    padding=self.padding)
        
        self.spade_right = SPADE(self.input_channels, self.kernel_size, 128, False)

        # the last layer
        '''self.conv = nn.Conv2d(self.input_channels,
                                self.output_channels,
                                self.kernel_size,
                                padding=self.padding)'''

        #self.upscale = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, input):

        z, x, mask = input
        
        # left branch
        left = self.spade_left((z, x, mask))
        left = F.leaky_relu(left, 0.2, inplace=True)
        left = self.conv_left(left)
        #left = F.leaky_relu(left, 0.2, inplace=True)

        # right branch
        right = self.spade_right((z, x, 1-mask))
        right = F.leaky_relu(right, 0.2, inplace=True)
        right = self.conv_right(right)
        #right = F.leaky_relu(right, 0.2, inplace=True)
        
        '''
        left = torch.cat([left, right], dim=1)
        left = F.leaky_relu(left, 0.2, inplace=True)
        
        left = F.pixel_shuffle(left, 2)
        left = self.conv(left)'''

        #left = left + right
        left = F.pixel_shuffle(left, 2)
        right = F.pixel_shuffle(right, 2)
        left = left + right
        #left = F.leaky_relu(left, 0.2, inplace=True)

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
