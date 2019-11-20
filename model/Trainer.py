import torch
from torch.autograd import Variable
from Losses import roi_loss, FeatureMatching
from torch.nn.parallel.scatter_gather import gather
import numpy as np

def check_grads(model, model_name):
    for p in model.parameters():
        if not p.grad is None:
            print(model_name, p.grad.size(), float(p.grad.mean()))

class Trainer:
    def __init__(self, models, optimizers, losses, clip_norm,
        writer, num_updates, device, multi_gpu, is_fmatch, n_layers_fe_matching, 
        is_roi_loss, is_wgan, wgan_clip_size):

        self.device = device
        self.multi_gpu = multi_gpu
        self.n_layers_fe_matching = n_layers_fe_matching

        self.gen, self.dis = models

        self.g_optimizer, self.d_optimizer = optimizers
        self.g_loss, self.d_loss = losses

        self.clip_norm = clip_norm
        self.writer = writer
        self.num_updates = num_updates

        self.is_fmatch = is_fmatch
        self.is_roi_loss = is_roi_loss

        self.wgan_clip_size = wgan_clip_size
        self.is_wgan = is_wgan

        self.n_devices = torch.cuda.device_count()

        if self.is_fmatch:

            self.fe_loss = FeatureMatching()
            self.extract_features = False

            for idx in self.n_layers_fe_matching:
                    
                name = 'temporary_{}'.format(str(idx))
                setattr(self, name, [])
                
                def _hook(name):
                    def hook(module, input, output):
                        if self.extract_features:
                            getattr(self, name).append(output)
                        else:
                            pass
                    return hook

                hook = _hook(name)

                if self.multi_gpu:
                    self.dis.module.net[idx].register_forward_hook(hook)
                else:
                    self.dis.net[idx].register_forward_hook(hook)

    def train_step_discriminator(self, noise, mask, batch):

        self.num_updates += 0.5
        
        # Discriminator
        self.dis.train()
        #self.gen.eval()
        self.d_optimizer.zero_grad()

        self.extract_features = False
            
        _, _, generated_samples = self.gen((noise, batch, mask))
        probs_fake = self.dis((generated_samples.detach(), mask))
        probs_real = self.dis((batch, mask))

        loss_d = self.d_loss(probs_fake, probs_real)

        return generated_samples, loss_d

    def train_step_generator(self, noise, mask, batch):

        self.num_updates += 0.5

        # Generator
        self.gen.train()
        #self.dis.eval()
        self.g_optimizer.zero_grad()

        self.extract_features = False

        mean, logvar, generated_samples = self.gen((noise, batch, mask))
        probs_fake = self.dis((generated_samples, mask)).detach()

        # or with detach?
        loss_g = self.g_loss(probs_fake)
        # vae loss
        loss_g += -torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

        if self.is_fmatch:

            self.extract_features = True
            # get internal features from D(x) and D(G(z))

            # initialize feats
            fake_feats = []
            real_feats = []

            # clear temporary variables
            for idx in self.n_layers_fe_matching:
                name = 'temporary_{}'.format(str(idx))
                setattr(self, name, [])

            # inference on the real batch
            _ = self.dis((batch, mask))

            # collect all intermidiate variables
            # per one layer for real batch
            for idx in self.n_layers_fe_matching:
                name = 'temporary_{}'.format(str(idx))
                if self.multi_gpu:
                    real_feats.append(gather(getattr(self, name), 0, 0))
                else:
                    real_feats.append(getattr(self, name)[0])

            # clear temporary variables
            for idx in self.n_layers_fe_matching:
                name = 'temporary_{}'.format(str(idx))
                setattr(self, name, [])

            # inference on the fake batch
            _ = self.dis((generated_samples, mask))

            # collect all intermidiate variables
            # per one layer for fake batch
            for idx in self.n_layers_fe_matching:
                name = 'temporary_{}'.format(str(idx))
                if self.multi_gpu:
                    fake_feats.append(gather(getattr(self, name), 0, 0))
                else:
                    fake_feats.append(getattr(self, name)[0])

            # clear all
            for idx in self.n_layers_fe_matching:
                name = 'temporary_{}'.format(str(idx))
                setattr(self, name, [])

            loss_g += self.fe_loss(fake_feats, real_feats) 

        if self.is_roi_loss:

            loss_roi = 1e-1 * ((1 - mask) * (generated_samples - batch))**2
            #loss_roi = (generated_samples - batch)**2
            loss_roi = loss_roi.mean() 

            loss_g += loss_roi

            #loss_int = (mask * (generated_samples - batch))**2
            #loss_int = 0.01 * loss_int.mean()

            #loss_g += loss_int

        return generated_samples, loss_g

    def backward_discriminator(self, loss_d):

        loss_d.backward()
        #check_grads(self.dis, 'dis')
        #torch.nn.utils.clip_grad_norm_(self.dis.parameters(), 1e-2)
        self.d_optimizer.step()
        if self.is_wgan:
            for parameters in self.dis.parameters():
                parameters.data.clamp_(-self.wgan_clip_size, self.wgan_clip_size)

    def backward_generator(self, loss_g):

        loss_g.backward()
        #check_grads(self.gen, 'gen')
        #torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1e-2)
        self.g_optimizer.step()
    #    loss.backward()
    #    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
    #    self.optimizer.step()
