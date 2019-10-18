import numpy as np
import argparse
import os
import json

import torch
from torch.autograd import Variable
from torch.optim import Adam
#from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from tensorboardX import SummaryWriter
#from torch.utils.data import DataLoader

from Models import Generator, Discriminator
#from Trainer import Trainer
from Data import Data
from torch.utils.data import DataLoader

from torchvision.utils import save_image

from utils import weights_init

parser = argparse.ArgumentParser()
parser.add_argument('--train_images', type=str, default='../cars_train')
#parser.add_argument('--masksdir', type=str, default='data/train/masks/')
#parser.add_argument('--testimages', type=str, default='data/test/images')
parser.add_argument('--logdir', type=str, default='data/logs')
parser.add_argument('--chkpdir', type=str, default='data/chkp')
parser.add_argument('--chkpname', type=str, default='none')
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--train_batch_size', type=int, default=64)
#parser.add_argument('--test_batch_size', type=int, default=100)
#parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
#parser.add_argument('--clip_norm', type=float, default=0.1)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creating dataloaders
obj = Data()
obj.download(args.train_images)
train_data = obj.data
data_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

#print('Number of samples for train - {}'.format(len(train_dataset)))
#print('Number of samples for test - {}'.format(len(test_dataset)))
#print('Train batch size - {}'.format(args.train_batch_size))
#print('Test batch size - {}'.format(args.test_batch_size))

# optimizer
lr = args.lr

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(args.n_epochs):
    for i, (imgs, _) in enumerate(data_loader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        z = Variable(Tensor(np.random.randn(64, 100, 1, 1)))
        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.n_epochs, i, len(data_loader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(data_loader) + i
        if batches_done % args.sample_interval == 0:
            save_image(gen_imgs.data[:25], "../generated/%d.png" % batches_done, nrow=5, normalize=True)
