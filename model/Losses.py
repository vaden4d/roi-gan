import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import numpy as np 

def tv_loss(img, tv_weight, mode='abs'):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    if mode == 'abs':
        w_variance = torch.mean((img[:,:,:,:-1] - img[:,:,:,1:]).abs())
        h_variance = torch.mean((img[:,:,:-1,:] - img[:,:,1:,:]).abs())
    elif mode == 'square':
        w_variance = torch.mean(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
        h_variance = torch.mean(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    else:
        raise NotImplementedError
    loss = tv_weight * (h_variance + w_variance)
    return loss

def roi_loss(fake, real):
    '''The loss function regulates'''

    loss = fake - real
    loss = torch.mean(loss**2)

    return loss

def vanilla_generator_loss(fake_outputs_probs):
    eps = 1e-12
    # log (1 - D(G(z))) -> min w.r.t G - standard setting
    # -log D(G(z)) -> min w.r.t G - Goodfellow recommendation
    #loss = (1 - fake_outputs_probs).log().mean()
    loss = -(fake_outputs_probs + eps).log().mean()
    return loss

def vanilla_discriminator_loss(fake_outputs_probs, real_outputs_probs):
    eps = 1e-12
    # -log D(x) - log (1 - D(G(z))) -> min w.r.t D
    loss = (1 - fake_outputs_probs + eps).log().mean() + 0.9 * (real_outputs_probs + eps).log().mean()
    return -loss

def softplus_generator_loss(fake_outputs_logprobs):
    loss = F.softplus(-fake_outputs_logprobs).mean()
    return loss

def softplus_discriminator_loss(fake_outputs_logprobs, real_outputs_logprobs):
    loss = F.softplus(fake_outputs_logprobs) + F.softplus(-real_outputs_logprobs)
    loss = loss.mean()
    return loss

def ls_generator_loss(fake_outputs_probs):
    # (1 - D(G(z)))**2 -> min w.r.t G - least squares setting
    loss = (1 - fake_outputs_probs)**2
    loss = loss.mean()
    return loss

def ls_discriminator_loss(fake_outputs_probs, real_outputs_probs):
    # (D(x)-1)**2 + (D(G(z)))**2 -> min w.r.t D
    loss = (real_outputs_probs - 1)**2 + fake_outputs_probs**2
    loss = loss.mean()
    return loss

def wasserstein_discriminator_loss(fake_outputs_logprobs, real_outputs_logprobs):
    # mean x D(x) - mean z D(G(z)), where D is dense vector -> max
    loss = -(real_outputs_logprobs.mean() - fake_outputs_logprobs.mean())
    return loss

def wasserstein_generator_loss(fake_outputs_logprobs):
    # mean z D(G(z)) -> max
    loss = -fake_outputs_logprobs.mean()
    return loss

def wasserstein_discriminator_hinge_loss(fake_outputs_logprobs, real_outputs_logprobs):
    # -mean x min(0, D(x)-1) - mean z min(0, -D(G(z))-1) -> min
    loss = F.relu(1-real_outputs_logprobs).mean() + F.relu(fake_outputs_logprobs+1).mean()
    return loss

class DiscriminatorLoss(nn.Module):

    def __init__(self, **kwargs):
        super(DiscriminatorLoss, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.loss == 'ls':
            self.loss_func = ls_discriminator_loss
        elif self.loss == 'vanilla':
            self.loss_func = vanilla_discriminator_loss
        elif self.loss == 'wgan':
            self.loss_func = wasserstein_discriminator_loss
        elif self.loss == 'softplus':
            self.loss_func = softplus_discriminator_loss
        elif self.loss == 'hinge':
            self.loss_func = wasserstein_discriminator_hinge_loss
        else:
            raise NotImplementedError

    def forward(self, fakes, reals):

        loss = self.loss_func(fakes, reals)
        
        return loss

class GeneratorLoss(nn.Module):

    def __init__(self, **kwargs):
        super(GeneratorLoss, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.loss == 'vanilla':
            self.loss_func = vanilla_generator_loss
        elif self.loss == 'ls':
            self.loss_func = ls_generator_loss
        elif self.loss == 'wgan':
            self.loss_func = wasserstein_generator_loss
        elif self.loss == 'softplus':
            self.loss_func = softplus_generator_loss
        elif self.loss == 'hinge':
            self.loss_func = wasserstein_generator_loss
        else:
            raise NotImplementedError

    def forward(self, fakes):
        loss = self.loss_func(fakes)
        return loss

class InfoLoss(nn.Module):
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """
    def __init__(self):
        super(InfoLoss, self).__init__()

    def forward(self, x, mu, var):
        
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll

class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()

    def gram_matrix(self, features):
    
        N, C, H, W = features.size()
        # Use torch.bmm for batch multiplication of matrices
        feat_reshaped = features.view(N, C, -1)
        gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
        gram = gram / (H*W*C)

        return gram

    def forward(self, input):
        
        x, y = input
        loss = (self.gram_matrix(x) - self.gram_matrix(y)).abs().mean(axis=[1, 2])

        return loss

class VGGLoss(nn.Module):
    def __init__(self, requires_grad=False, weight_closs=5, weight_sloss=5):
        super(VGGLoss, self).__init__()
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

        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.style_loss = StyleLoss()

        self.content_weight = weight_closs
        self.style_weight = weight_sloss

    def forward(self, input):
        # x - generated
        # y - completed
        # z - real
        x, y, z = input
        loss = 0
        for i in range(len(self.weights)):

            x = getattr(self, 'slice{}'.format(i+1))(x)
            y = getattr(self, 'slice{}'.format(i+1))(y)
            z = getattr(self, 'slice{}'.format(i+1))(z)
            #loss += self.weights[i] * (self.criterion(x, z) + self.criterion(y, z))
            loss += self.content_weight * self.weights[i] * (x - z).abs().mean(axis=[1, 2, 3])
            loss += self.content_weight * self.weights[i] * (y - z).abs().mean(axis=[1, 2, 3])

            #loss += self.style_weight * self.weights[i] * self.style_loss((x, z))
            #loss += self.style_weight * self.weights[i] * self.style_loss((y, z))
        #loss = loss.view(1, 1)
        
        return loss

class FeatureMatching(nn.Module):

    def __init__(self):
        super(FeatureMatching, self).__init__()

    def forward(self, x, y):

        assert type(x) == type(y)
        assert len(x) == len(y)
        
        loss = 0
        #weights = list(range(1, len(x)+1))
        #weights = list(map(lambda z: z / len(x), weights))

        for tensor_x, tensor_y in zip(x, y):

            loss += (tensor_x - tensor_y).abs().mean()

        return loss
