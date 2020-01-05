import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

class Conv2d(nn.Module):
    '''Equalized Learning Rate convolution
    layer'''
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super(Conv2d, self).__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, stride=self.stride, padding=self.padding)
        else:
            return F.conv2d(x, self.weight * self.w_lrmul, stride=self.stride, padding=self.padding)

class Linear(nn.Module):
    '''Equalized Learning Rate linear
    layer'''
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(Linear, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        return out

class ConvTranspose2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super(ConvTranspose2d, self).__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(input_channels, output_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv_transpose2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, stride=self.stride, padding=self.padding)
        else:
            return F.conv_transpose2d(x, self.weight * self.w_lrmul, stride=self.stride, padding=self.padding)

class GatedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, 
                stride, padding=0, bias=True, normalization=True):
        super(GatedConv2d, self).__init__()
        self.conv2d = Conv2d(in_channels, out_channels, kernel_size, 
                            stride=stride, padding=padding, bias=bias)
        self.mask_conv2d = Conv2d(in_channels, out_channels, kernel_size, 
                            stride=stride, padding=padding, bias=bias)
        self.normalization = normalization
        if self.normalization:
            self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):

        feats = self.conv2d(x)
        feats = F.leaky_relu(feats, 0.2, inplace=True)
        
        mask = self.mask_conv2d(x)
        mask = F.sigmoid(mask)

        output = mask * feats
        if self.normalization:
            output = self.norm(output)

        return output

class GatedConv2dWithSpectral(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, 
                stride, padding=0, bias=True, normalization=True):
        super(GatedConv2dWithSpectral, self).__init__()
        self.conv2d = spectral_norm(Conv2d(in_channels, out_channels, kernel_size, 
                            stride=stride, padding=padding, bias=bias))
        self.mask_conv2d = spectral_norm(Conv2d(in_channels, out_channels, kernel_size, 
                            stride=stride, padding=padding, bias=bias))
        self.normalization = normalization
        if self.normalization:
            self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):

        feats = self.conv2d(x)
        feats = F.leaky_relu(feats, 0.2, inplace=True)
        
        mask = self.mask_conv2d(x)
        mask = F.sigmoid(mask)

        output = mask * feats
        if self.normalization:
            output = self.norm(output)

        return output