import torch
import torch.nn as nn

def roi_loss(masks, real, fake):
    '''The loss function regulates'''

    loss = (1 - masks) * fake - real
    loss = torch.mean(loss**2)

    return loss

def vanilla_generator_loss(fake_outputs_probs):
    # -log D(G(z)) -> min w.r.t G
    loss = -fake_outputs_probs.log().mean()
    return loss

def vanilla_discriminator_loss(fake_outputs_probs, real_outputs_probs):
    # -log D(x) - log D(1 - G(z)) -> min w.r.t D
    loss = (1 - fake_outputs_probs).log().mean() + real_outputs_probs.log().mean()
    return -loss
