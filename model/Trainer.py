import torch
from torch.autograd import Variable
from Losses import roi_loss, FeatureMatching
from torch.nn.parallel.scatter_gather import gather

class Trainer:
    def __init__(self, models, optimizers, losses, clip_norm,
        writer, num_updates, device, multi_gpu, is_fmatch, n_layers_fe_matching, is_roi_loss):

        self.device = device
        self.multi_gpu = multi_gpu
        self.n_layers_fe_matching = n_layers_fe_matching
        '''
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            self.gen, self.dis = models[0].to(self.device), models[1].to(self.device)
        else:
            self.gen, self.dis = models

        if self.multi_gpu:
            self.gen = torch.nn.DataParallel(self.gen)
            self.dis = torch.nn.DataParallel(self.dis)'''

        self.gen, self.dis = models

        self.g_optimizer, self.d_optimizer = optimizers
        self.g_loss, self.d_loss = losses

        self.clip_norm = clip_norm
        self.writer = writer
        self.num_updates = num_updates
        self.is_fmatch = is_fmatch
        self.is_roi_loss = is_roi_loss

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
        self.gen.eval()
        self.d_optimizer.zero_grad()

        self.extract_features = False
            
        #_, _, generated_samples = self.gen((noise, batch, mask))
        generated_samples = self.gen((noise, batch, mask))
        probs_fake = self.dis((generated_samples, mask))
        probs_real = self.dis((batch, mask))

        self.loss_d = self.d_loss(probs_fake, probs_real)
        #self.loss_d.backward()
        #self.d_optimizer.step()

        return generated_samples, self.loss_d

    def train_step_generator(self, noise, mask, batch):

        self.num_updates += 0.5

        # Generator
        self.gen.train()
        self.dis.eval()
        self.g_optimizer.zero_grad()

        self.extract_features = False

        #mean, logvar, generated_samples = self.gen((noise, batch, mask))
        generated_samples = self.gen((noise, batch, mask))
        probs_fake = self.dis((generated_samples, mask))

        # or with detach?
        self.loss_g = self.g_loss(probs_fake)
        # vae loss
        #self.loss_g += -0.1 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

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

            self.loss_g += self.fe_loss(fake_feats, real_feats) / len(self.n_layers_fe_matching)

        if self.is_roi_loss:

            loss_roi = ((1 - mask) * (generated_samples - batch))**2
            loss_roi = loss_roi.mean() 

            self.loss_g += loss_roi

            #loss_int = (mask * (generated_samples - batch))**2
            #loss_int = 0.1 * loss_int.mean()

            #self.loss_g += loss_int

        return generated_samples, self.loss_g

    def backward_discriminator(self):

        self.loss_d.backward()
        #torch.nn.utils.clip_grad_norm_(self.dis.parameters(), 1e-2)
        self.d_optimizer.step()

    def backward_generator(self):

        self.loss_g.backward()
        #torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1e-2)
        self.g_optimizer.step()
    #    loss.backward()
    #    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
    #    self.optimizer.step()
