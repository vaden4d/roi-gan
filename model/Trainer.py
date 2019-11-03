import torch
from torch.autograd import Variable
from Losses import roi_loss

class Trainer:
    def __init__(self, models, optimizers, losses, clip_norm,
        writer, num_updates, device, multi_gpu, is_fmatch, is_roi_loss):

        self.device = device

        if self.device.type == 'cuda':
            self.gen, self.dis = models[0].cuda(), models[1].cuda()
        else:
            self.gen, self.dis = models

        if multi_gpu:
            self.gen = torch.nn.DataParallel(self.gen)
            self.dis = torch.nn.DataParallel(self.dis)

        self.g_optimizer, self.d_optimizer = optimizers
        self.g_loss, self.d_loss = losses

        self.clip_norm = clip_norm
        self.writer = writer
        self.num_updates = num_updates
        self.is_fmatch = is_fmatch
        self.is_roi_loss = is_roi_loss

    def train_step_discriminator(self, noise, mask, batch):

        self.num_updates += 0.5
        
        # Discriminator
        self.dis.train()
        self.gen.eval()
        self.d_optimizer.zero_grad()
            
        generated_samples = self.gen(noise, mask)
        probs_fake = self.dis(generated_samples, mask)
        probs_real = self.dis(batch, mask)

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

        generated_samples = self.gen(noise, mask)
        probs_fake = self.dis(generated_samples, mask)

        # or with detach?
        self.loss_g = self.g_loss(probs_fake)

        if self.is_fmatch:
            
            # get internal features from D(x) and D(G(z))
            fake_feats = self.dis.int_outputs
            _ = self.dis(batch, mask)
            real_feats = self.dis.int_outputs

            for fake_feat, real_feat in zip(fake_feats, real_feats):

                loss_mse = (fake_feat - real_feat)**2
                loss_mse = loss_mse.mean()
                self.loss_g += loss_mse / len(self.dis.int_outputs)

            
        
        if self.is_roi_loss:

            loss_roi = ((1 - mask) * (generated_samples - batch))**2
            loss_roi = loss_roi.mean() 

            self.loss_g += loss_roi
        #loss_roi = ((masks - batch)**2).mean()
        #(loss_g + loss_roi).backward()
        #self.loss_g.backward()
        #self.g_optimizer.step()

        return generated_samples, self.loss_g

    def backward_discriminator(self):

        self.loss_d.backward()
        self.d_optimizer.step()

    def backward_generator(self):

        self.loss_g.backward()
        self.g_optimizer.step()
    #    loss.backward()
    #    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
    #    self.optimizer.step()
