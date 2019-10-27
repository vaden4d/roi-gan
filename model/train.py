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

from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from Models import Generator, Discriminator
from Trainer import Trainer
from utils.rois import RoI, gaussian_roi
from utils.functions import weights_init

from Data import Data
from torch.utils.data import DataLoader
from Losses import vanilla_generator_loss, vanilla_discriminator_loss
from torchvision.utils import save_image

from tensorboardX import SummaryWriter

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

dataset_name = config.dataset

image_size = config.datasets_hyperparams[dataset_name]['img_shape']
data_path = config.datasets_hyperparams[dataset_name]['path']
mean = config.datasets_hyperparams[dataset_name]['mean']
std = config.datasets_hyperparams[dataset_name]['std']

# creating dataloaders
obj = Data(data_path)
train_data = obj.data
data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

print('Number of samples for train - {}'.format(len(data_loader)))
print('Batch size - {}'.format(batch_size))

# Initialize generator, discriminator and RoI generator
generator = Generator()
discriminator = Discriminator()
roi = RoI(image_size, gaussian_roi, device)

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
if chkpdir and chkpname_dis and chkpname_gen:

    # load state of networks
    state_gen = load_model(chkpdir, chkpname_gen)
    state_dis = load_model(chkpdir, chkpname_dis)

    # load generator
    generator.load_state_dict(state_gen['model'])
    generator = generator.to(device)
    optimizer_G.load_state_dict(state_gen['optimizer'])

    # load discriminator
    discriminator.load_state_dict(state_dis['model'])
    discriminator = discriminator.to(device)
    optimizer_D.load_state_dict(state_dis['optimizer'])

    initial_epoch = state['epoch']
    num_updates = state['iter']

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

writer = SummaryWriter(logdir)
trainer = Trainer(models=[generator,
                            discriminator],
                    optimizers=[optimizer_G,
                                optimizer_D],
                    losses=[vanilla_generator_loss,
                            vanilla_discriminator_loss],
                    clip_norm=clip_norm,
                    writer=writer,
                    num_updates=num_updates,
                    device=device,
                    multi_gpu=multi_gpu)

num_batches = len(data_loader)
# train and evaluate
for epoch in range(0, num_epochs):

    # train loop
    train_loss_gen = 0
    train_loss_dis = 0

    current_epoch = initial_epoch + epoch
    with tqdm(ascii=True, leave=False,
              total=len(data_loader), desc='Epoch {}'.format(current_epoch)) as bar:

        for images in train_loader:

            # gradient update
            images = images.to(device)

            random = Variable(Tensor(np.random.randn(batch_size, 100, 1, 1)))
            masks = roi.generate_masks(batch_size)
            masks = images * (1 - masks)
            gen_images, loss_d, loss_g = trainer.train_step(random, masks, imgs)

            # compute loss and accuracy
            train_loss_gen += loss_g.item()
            train_loss_dis += loss_d.item()

            bar.postfix = 'loss D - {:.5f}, loss G - {:.5f}, lr - {:.5f}'.format(
                                                                    loss_d
                                                                    loss_g,
                                                                    lr
                                                                   )
            bar.update()

            trainer.writer.add_scalars('iter_loss/loss_d', {'train' : loss_d.item()}, trainer.num_updates)
            trainer.writer.add_scalars('iter_loss/loss_g', {'train' : loss_g.item()}, trainer.num_updates)

    # log train stats
    train_loss_gen /= num_batches
    train_loss_dis /= num_batches

    trainer.writer.add_scalars('epoch_loss/loss_d', {'train' : train_loss_dis}, trainer.num_updates)
    trainer.writer.add_scalars('epoch_loss/loss_g', {'train' : train_loss_gen}, trainer.num_updates)

    print('Epoch {}: Loss D - {:.5f}, Loss G - {:.5f}'.format(current_epoch, train_loss_dis, train_loss_gen))

    # save model
    save_model(trainer.gen, trainer.g_optimizer, current_epoch, trainer.num_updates, chkpdir, 'gen')
    save_model(trainer.dis, trainer.d_optimizer, current_epoch, trainer.num_updates, chkpdir, 'dis')

    if epoch % sample_interval == 0:

        save_image(gen_images.data[:25], 'generated/%d.png' % current_epoch, nrow=5, normalize=True)
