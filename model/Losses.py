import torch
import torch.nn as nn

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

class DiscriminatorLoss(nn.Module):

    def __init__(self, **kwargs):
        super(DiscriminatorLoss, self).__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.loss == 'ls':
            self.loss_func = ls_discriminator_loss
        elif self.loss == 'vanilla':
            self.loss_func = vanilla_discriminator_loss
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
        else:
            raise NotImplementedError

    def forward(self, fakes):
        loss = self.loss_func(fakes)
        return loss

'''
class FeatureMatching(nn.Module):

    def __init__(self):
        super(FeatureMatching, self).__init__()
    
    def forward(self, x, y):'''

class FeatureMatching(nn.Module):

    def __init__(self):
        super(FeatureMatching, self).__init__()

    def forward(self, x, y):

        assert type(x) == type(y)
        assert len(x) == len(y)
        
        loss = 0

        for tensor_x, tensor_y in zip(x, y):

            loss += ((tensor_x - tensor_y)**2).mean()

        return loss
