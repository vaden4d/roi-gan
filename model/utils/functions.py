import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image

ef save_model(model, optimizer, epoch, iter, chkp_dir, name):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'iter': iter
    }
    torch.save(state, os.path.join(chkp_dir, '{}-epoch-{}.chkp'.format(name, epoch)))


def load_model(chkp_dir, chkp_name):
    state = torch.load(os.path.join(chkp_dir, chkp_name))
    return state

def weights_init(m, init_type='normal', gain=0.02):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':
            m.reset_parameters()
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

def parameters_summary(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network [%s] summary:\nTotal number of parameters: %.1fM.'
              % (type(net).__name__, num_params / 1000000))


if __name__ == '__main__':

    pass
