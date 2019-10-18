
import torch
import torch.nn as nn
import torch.nn.functional as F

class SPADE(nn.Module):

    def __init__(self, n_channels, kernel_size):
        super().__init__()

        self.normalization = nn.InstanceNorm2d(n_channels, affine=False)

        # hidden layer
        n_hidden = 128
        pw = kernel_size // 2
        self.dense = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU()
        )

        self.dense_gamma = nn.Conv2d(n_hidden, n_channels, kernel_size=kernel_size, padding=pw)
        self.dense_beta = nn.Conv2d(n_hidden, n_channels, kernel_size=kernel_size, padding=pw)

    def forward(self, x, mask):

        # normalize input
        normalized = self.normalization(x)

        # beta and gamma generation
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')

        activations = self.dense(mask)
        gamma = self.dense_gamma(activations)
        beta = self.dense_beta(activations)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

if __name__ == '__main__':

    x = torch.randn(10, 5, 30, 30)
    mask = torch.randn(10, 1, 13, 13)
    obj = SPADE(5, 3)
    z = obj(x, mask)
    print(z.size())
