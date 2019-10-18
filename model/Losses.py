import torch
import torch.nn as nn

def roi_loss(masks, x_real, x_fake):
    '''The loss function regulates'''

    loss = (1 - masks) * x_fake - x_real
    loss = torch.mean(loss**2)

    return loss
