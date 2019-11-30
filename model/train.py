import numpy as np
import argparse
from tqdm import tqdm
import config

gpu = config.train_mode['gpu']
visible_devices = config.train_mode['gpu_devices'] if gpu else "-1"
multi_gpu = config.train_mode['multi_gpu'] and gpu
print('Training with {}:'.format('GPU' if gpu else 'CPU'))
if multi_gpu:
    print('Multi-GPU mode is enabled: used {} gpus.'.format(visible_devices))

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

import torchvision
import torchvision.transforms as transforms

import torch
from torch.nn import DataParallel
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from Models import Generator, Discriminator
from Trainer import Trainer
from utils.rois import RoI, gaussian_roi, squared_roi, mixture_roi
from utils.functions import weights_init, save_model, load_model

from Data import Data
from torch.utils.data import DataLoader
from Losses import GeneratorLoss, DiscriminatorLoss
from torchvision.utils import save_image

from tensorboardX import SummaryWriter
from torchsummary import summary

device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

logdir = config.logs_hyperparams['log_dir']
chkpdir = config.logs_hyperparams['chkp_dir']
chkpname_gen = config.logs_hyperparams['chkp_name_gen']
chkpname_dis = config.logs_hyperparams['chkp_name_dis']

try:
    os.mkdir(chkpdir)
except FileExistsError:
    pass

# get hyperparameters from static file
num_epochs = config.train_hyperparams['num_epochs']
batch_size = config.train_hyperparams['batch_size']
sample_interval = config.train_hyperparams['sample_interval']

lr_gen = config.optimizator_hyperparams['lr_gen']
lr_dis = config.optimizator_hyperparams['lr_dis']

clip_norm = config.model_hyperparams['clip_norm']
roi_function = config.model_hyperparams['function']

print_summary = config.print_summary

gen_input_channels = config.gen_hyperparams['input_channels']
input_size = config.gen_hyperparams['init_size']

dis_n_features = config.model_hyperparams['dis_n_features']

dataset_name = config.dataset

image_size = config.datasets_hyperparams[dataset_name]['img_shape']
data_path = config.datasets_hyperparams[dataset_name]['path']
mean = config.datasets_hyperparams[dataset_name]['mean']
std = config.datasets_hyperparams[dataset_name]['std']

is_add_noise = config.stabilizing_hyperparams['adding_noise']

is_fe_matching = config.discriminator_stabilizing_hyperparams['fe_matching']
n_layers_fe_matching = config.discriminator_stabilizing_hyperparams['n_layers_fe_matching']
is_roi_loss = config.generator_stabilizing_hyperparams['roi_loss']

# creating dataloaders
train_data = Data(data_path, mean, std)
data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=5)

print('Number of samples for train - {}'.format(len(data_loader)))
print('Batch size - {}'.format(batch_size))

generator_loss_hyperparams = config.generator_stabilizing_hyperparams
discriminator_loss_hyperparams = config.discriminator_stabilizing_hyperparams

# wgan setting
if (generator_loss_hyperparams['loss'] == 'wgan' and discriminator_loss_hyperparams['loss'] != 'wgan') or \
    (generator_loss_hyperparams['loss'] != 'wgan' and discriminator_loss_hyperparams['loss'] == 'wgan'):
    raise NotImplementedError

wgan_clip_size = discriminator_loss_hyperparams['wgan_clip_size']
is_wgan = generator_loss_hyperparams['loss'] == 'wgan' and discriminator_loss_hyperparams == 'wgan'

# Initialize generator, discriminator and RoI generator
generator = Generator(**config.gen_hyperparams)
discriminator = Discriminator(dis_n_features, is_wgan=is_wgan)
roi = RoI(image_size, locals()[roi_function], len(train_data))
roi_loader = DataLoader(roi, batch_size=batch_size, shuffle=False, num_workers=5)

from utils.functions import count_parameters
if print_summary:
    print('Generator:')
    count_parameters(generator)
    #summary(generator, [(gen_n_input, 1, 1), (1, 64, 64)], device='cpu')
    print('Discriminator')
    count_parameters(discriminator)
    #summary(discriminator, (3, 64, 64), device='cpu')

# Initialize weights
if chkpname_dis == None and chkpname_gen == None:
    generator.apply(weights_init)
    discriminator.apply(weights_init)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_gen, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_dis, betas=(0.5, 0.999))

initial_epoch = 0
num_updates = 0
# if there is checkpoint, download it

if device.type == 'cuda':

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    discriminator = discriminator.to(device)
    generator = generator.to(device)

if multi_gpu:

    generator = torch.nn.DataParallel(generator)
    discriminator = torch.nn.DataParallel(discriminator)

if chkpdir and chkpname_dis and chkpname_gen:

    # load state of networks
    state_gen = load_model(chkpdir, chkpname_gen)
    state_dis = load_model(chkpdir, chkpname_dis)

    # load generator
    #generator = generator.to(device)
    if multi_gpu:
        #generator = DataParallel(generator)
        generator = generator.cuda()
    generator.load_state_dict(state_gen['model'])
    #generator = generator.to(device)
    optimizer_G.load_state_dict(state_gen['optimizer'])

    # load discriminator
    #discriminator = discriminator.to(device)
    if multi_gpu:
        #discriminator = DataParallel(discriminator)
        discriminator = discriminator.cuda()
    discriminator.load_state_dict(state_dis['model'])
    #discriminator = discriminator.to(device)
    optimizer_D.load_state_dict(state_dis['optimizer'])

    initial_epoch = state_gen['epoch']
    num_updates = state_gen['iter']

generator_loss = GeneratorLoss(**generator_loss_hyperparams)
discriminator_loss = DiscriminatorLoss(**discriminator_loss_hyperparams)

writer = SummaryWriter(logdir)
trainer = Trainer(models=[generator,
                            discriminator],
                    optimizers=[optimizer_G,
                                optimizer_D],
                    losses=[generator_loss,
                            discriminator_loss],
                    clip_norm=clip_norm,
                    writer=writer,
                    num_updates=num_updates,
                    device=device,
                    multi_gpu=multi_gpu,
                    is_fmatch=is_fe_matching,
                    n_layers_fe_matching=n_layers_fe_matching,
                    is_roi_loss=is_roi_loss,
                    is_wgan=is_wgan,
                    wgan_clip_size=wgan_clip_size
                    )

# saving generated
try:
    os.mkdir('generated')
except FileExistsError:
    pass

train_dis = True
train_gen = True

num_batches = len(data_loader)
# train and evaluate
for epoch in range(0, num_epochs):

    # train loop
    train_loss_gen = 0
    train_loss_dis = 0

    i = 0
    current_epoch = initial_epoch + epoch
    with tqdm(ascii=True, leave=False,
              total=len(data_loader), desc='Epoch {}'.format(current_epoch)) as bar:

        for mask, images in zip(roi_loader, data_loader):

            images = images.to(device)
            mask = mask.unsqueeze(1).to(device)

            if is_add_noise:
                images += 0.05 * torch.randn(images.size()).to(device)
            # if final batch isn't equal to defined batch size in loader
            batch_size = images.size()[0]
            
            random = torch.randn(batch_size, 2*128).to(device)
            _, loss_d = trainer.train_step_discriminator(random, mask, images)

            random = torch.randn(batch_size, 2*128).to(device)
            gen_images, loss_g = trainer.train_step_generator(random, mask, images)
            '''
            if is_wgan:
                
                if loss_d.item() < -20.0 * (1+epoch) / num_epochs:
                    train_dis = False
                else:
                    train_dis = True
                pass
            else:

                if loss_d.item() < 0.4 * (1 - epoch / num_epochs):
                    train_dis = False
                else:
                    train_dis = True'''

            if train_dis:
                trainer.backward_discriminator(loss_d)

            if train_gen:
                trainer.backward_generator(loss_g)

            if i % sample_interval == 0:
        
                save_image(gen_images.data[:25], 
                            'generated/%d_%d.png' % (current_epoch, i), 
                            nrow=5, normalize=True)
            
            #if loss_d.item() < 0.3 * (1 - epoch / num_epochs):
            #    train_dis = False
            #else:
            #    train_dis = True
            #train_gen = loss_g.item() * 1.5 > loss_d.item()
            #train_dis = loss_d.item() * 1.5 > loss_g.item()

            # compute loss and accuracy
            train_loss_gen += loss_g.item()
            train_loss_dis += loss_d.item()

            bar.postfix = 'loss D - {:.5f}, loss G - {:.5f}, lr D - {:.7f}, lr G - {:.7f}'.format(
                                                                    loss_d,
                                                                    loss_g,
                                                                    lr_dis,
                                                                    lr_gen
                                                                   )
            bar.update()

            trainer.writer.add_scalars('iter_loss/loss_d', {'train' : loss_d.item()}, trainer.num_updates)
            trainer.writer.add_scalars('iter_loss/loss_g', {'train' : loss_g.item()}, trainer.num_updates)

            # freed memory
            torch.cuda.empty_cache()

            i += 1

    # log train stats
    train_loss_gen /= num_batches
    train_loss_dis /= num_batches

    trainer.writer.add_scalars('epoch_loss/loss_d', {'train' : train_loss_dis}, trainer.num_updates)
    trainer.writer.add_scalars('epoch_loss/loss_g', {'train' : train_loss_gen}, trainer.num_updates)

    print('Epoch {}: Loss D - {:.5f}, Loss G - {:.5f}'.format(current_epoch, train_loss_dis, train_loss_gen))

    # save model
    save_model(trainer.gen, trainer.g_optimizer, current_epoch, trainer.num_updates, chkpdir, 'gen')
    save_model(trainer.dis, trainer.d_optimizer, current_epoch, trainer.num_updates, chkpdir, 'dis')
