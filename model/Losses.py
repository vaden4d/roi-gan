import torch
import torch.nn as nn

def roi_loss(masks, real, fake):
    '''The loss function regulates'''

    loss = (1 - masks) * fake - real
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