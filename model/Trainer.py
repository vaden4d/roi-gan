import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from Losses import roi_loss, FeatureMatching, VGGLoss, InfoLoss, tv_loss
from torch.nn.parallel.scatter_gather import gather
import numpy as np

def check_grads(model, model_name):
    for p in model.parameters():
        if not p.grad is None:
            print(model_name, p.grad.size(), float(p.grad.mean()))

class Trainer:
    def __init__(self, models, optimizers, losses, clip_norm,
        writer, num_updates, device, multi_gpu, is_fmatch, n_layers_fe_matching, 
        is_roi_loss, is_wgan, wgan_clip_size, 
        noise_dim,
        cont_dim,
        disc_dim,
        n_disc):

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

        self.vgg_loss = True

        self.noise_dim = noise_dim
        self.cont_dim = cont_dim
        self.disc_dim = disc_dim
        self.n_disc = n_disc

        self.info_loss = InfoLoss()
        self.disc_info_loss = nn.CrossEntropyLoss()

        if self.vgg_loss:

            self.vgg = VGGLoss()
            if self.multi_gpu:
                self.vgg = self.vgg.to(device)
                self.vgg = torch.nn.DataParallel(self.vgg)
            else:
                self.vgg = self.vgg.to(device)

        if self.is_fmatch:

            self.fe_loss = FeatureMatching()
            self.extract_features = False

            for idx in self.n_layers_fe_matching:
                    
                name = 'temporary_{}'.format(str(idx))
                setattr(self, name, [])
                
                def _hook(name):
                    def hook(module, input, output):
                        if self.extract_features:
                            getattr(self, name).append(output[0])
                        else:
                            pass
                    return hook

                hook = _hook(name)

                if self.multi_gpu:
                    self.dis.module.net[idx].register_forward_hook(hook)
                    #getattr(self.dis.module, 'block_{}'.format(idx)).register_forward_hook(hook)
                else:
                    self.dis.net[idx].register_forward_hook(hook)
                    #getattr(self.dis, 'block_{}'.format(idx)).register_forward_hook(hook)

    def train_step_discriminator(self, noise, mask, batch):

        self.num_updates += 0.5
        
        # Discriminator
        self.dis.train()
        self.gen.eval()

        if self.multi_gpu:
            for param in self.dis.module.net.parameters():
                param.requires_grad = True
            for param in self.dis.module.head_local.parameters():
                param.requires_grad = True
            for param in self.dis.module.head_global.parameters():
                param.requires_grad = True
            
            for param in self.gen.module.parameters():
                param.requires_grad = False
            for param in self.dis.module.q_net.parameters():
                param.requires_grad = False
        else:
            for param in self.dis.net.parameters():
                param.requires_grad = True
            for param in self.dis.head_local.parameters():
                param.requires_grad = True
            for param in self.dis.head_global.parameters():
                param.requires_grad = True
            
            for param in self.gen.parameters():
                param.requires_grad = False
            for param in self.dis.q_net.parameters():
                param.requires_grad = False

        self.d_optimizer.zero_grad()

        self.extract_features = False

        generated_samples, _, _, _ = self.gen((noise, batch, mask))
        #_, _, probs_fake = self.dis((generated_samples, mask))
        #_, _, probs_real = self.dis((batch, mask))
        completed_samples = mask * generated_samples + (1-mask) * batch
        
        probs_fake = self.dis((completed_samples, mask, True))
        probs_real = self.dis((batch, mask, True))

        loss_d = self.d_loss(probs_fake, probs_real)

        loss_d += 5 * self._gradient_penalty(batch, completed_samples, mask)

        loss_d += 1e-4 * (probs_real**2).mean()

        return generated_samples, loss_d

    def train_step_generator(self, noise, mask, batch):

        self.num_updates += 0.5

        # Generator
        self.gen.train()
        self.dis.eval()

        if self.multi_gpu:
            for param in self.dis.module.net.parameters():
                param.requires_grad = False
            for param in self.dis.module.head_local.parameters():
                param.requires_grad = False
            for param in self.dis.module.head_global.parameters():
                param.requires_grad = False
            
            for param in self.gen.module.parameters():
                param.requires_grad = True
            for param in self.dis.module.q_net.parameters():
                param.requires_grad = True
        else:
            for param in self.dis.net.parameters():
                param.requires_grad = False
            for param in self.dis.head_local.parameters():
                param.requires_grad = False
            for param in self.dis.head_global.parameters():
                param.requires_grad = False
            
            for param in self.gen.parameters():
                param.requires_grad = True
            for param in self.dis.q_net.parameters():
                param.requires_grad = True

        self.g_optimizer.zero_grad()

        self.extract_features = False

        generated_samples, mean_latent, logvar_latent, z = self.gen((noise, batch, mask))
        completed_samples = mask * generated_samples + (1-mask) * batch
        #with torch.no_grad():
            #probs_fake = self.dis((generated_samples, mask))
        mean, var, disc, probs_fake = self.dis((completed_samples, mask, False))

        # or with detach?
        loss_g = self.g_loss(probs_fake)

        # vae loss
        loss_g += -0.5 * torch.mean(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp())

        info_loss = 0.5 * self.info_loss(noise[:, self.noise_dim:self.noise_dim+self.cont_dim], mean, var)
        loss_g += info_loss
        for i in range(self.n_disc):
            loss_g += 0.5 * self.disc_info_loss(disc[:, i*self.disc_dim:(i+1)*self.disc_dim], 
                                          noise[:, self.noise_dim+self.cont_dim+i*self.disc_dim:self.noise_dim+self.cont_dim+(i+1)*self.disc_dim].argmax(axis=1))
        #print(info_loss.item())

        # total variation loss
        loss_g += tv_loss(mask * completed_samples, 0.5)
        #loss_g += tv_loss(generated_samples, 1e-1)

        if self.vgg_loss:
            loss_g += 5 * self.vgg((generated_samples, completed_samples, batch)).mean()

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
            _ = self.dis((batch, mask, True))

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
            _ = self.dis((completed_samples, mask, True))

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

            loss_g += 0.1 * self.fe_loss(fake_feats, real_feats) 

        if self.is_roi_loss:

            loss_roi = 1e-2 * (mask * (generated_samples - batch)).abs()
            #loss_roi = (generated_samples - batch)**2
            loss_roi = loss_roi.mean() 

            loss_g += loss_roi

            #loss_int = 0.3 * ((1-mask) * (generated_samples - batch)).abs()
            #loss_int = loss_int.mean()

            #loss_g += loss_int

        return completed_samples, loss_g, info_loss

    def backward_discriminator(self, loss_d):

        loss_d.backward()
        #check_grads(self.dis, 'dis')
        #torch.nn.utils.clip_grad_norm_(self.dis.parameters(), 1e-2)
        self.d_optimizer.step()
        if self.is_wgan and self.wgan_clip_size:
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

    def _gradient_penalty(self, real_data, generated_data, masks):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(self.device)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.dis((interpolated, masks, True))

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        
        #CHANGE MASK * GRADIENTS IF NOT WORKING
        gradients = gradients * (1-masks)
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon

        gradients_norm = torch.sqrt(torch.sum((gradients) ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()