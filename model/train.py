import numpy as np
import argparse
import os
from time import time

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from tensorboardX import SummaryWriter
#from torch.utils.data import DataLoader

from Models import Generator, Discriminator
from Trainer import Trainer
from utils.rois import RoI, gaussian_roi
from utils.functions import weights_init

from Data import Data
from torch.utils.data import DataLoader
from Losses import vanilla_generator_loss, vanilla_discriminator_loss
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--images', type=str, default='/Users/vaden4d/Documents/ds/roi-gan/cars_train')
#parser.add_argument('--logdir', type=str, default='data/logs')
#parser.add_argument('--chkpdir', type=str, default='data/chkp')
#parser.add_argument('--chkpname', type=str, default='none')
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
#parser.add_argument('--test_batch_size', type=int, default=100)
#parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
#parser.add_argument('--clip_norm', type=float, default=0.1)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creating dataloaders
obj = Data()
obj.download(args.images)
train_data = obj.data
data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

#print('Number of samples for train - {}'.format(len(train_dataset)))
#print('Number of samples for test - {}'.format(len(test_dataset)))
#print('Train batch size - {}'.format(args.train_))
#print('Test batch size - {}'.format(args.test_batch_size))

# optimizer
lr = args.lr

# Loss function
gen_loss = torch.nn.BCELoss()

# Initialize generator, discriminator and RoI generator
generator = Generator().to(device)
discriminator = Discriminator().to(device)
roi = RoI((64, 64), gaussian_roi, device)

# Initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

trainer = Trainer([generator, discriminator], [optimizer_G, optimizer_D],
                    [vanilla_generator_loss,
                    vanilla_discriminator_loss], device)

# ----------
#  Training
# ----------
batch_size = args.batch_size
for epoch in range(args.n_epochs):
    for i, (imgs, _) in enumerate(data_loader):

        start = time()

        random = Variable(Tensor(np.random.randn(batch_size, 100, 1, 1)))
        masks = roi.generate_masks(batch_size)
        masks = imgs * (1 - masks)
        gen_imgs, loss_d, loss_g = trainer.train_step(random, masks, imgs)

        t = time() - start

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Time %.3fs]"
            % (epoch, args.n_epochs, i, len(data_loader), loss_d, loss_g, t)
        )

        batches_done = epoch * len(data_loader) + i
        if batches_done % args.sample_interval == 0:
            save_image(gen_imgs.data[:25], "../generated/%d.png" % batches_done, nrow=5, normalize=True)
