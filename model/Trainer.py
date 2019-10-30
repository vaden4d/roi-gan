import torch
from torch.autograd import Variable
from Losses import roi_loss

class Trainer:
    def __init__(self, models, optimizers, losses, clip_norm,
        writer, num_updates, device, multi_gpu, is_fmatch):

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

    def train_step_discriminator(self, noise, mask, batch):

        self.num_updates += 0.5
        
        # Discriminator
        self.dis.train()
        self.gen.eval()
        self.d_optimizer.zero_grad()
            
        generated_samples = self.gen(noise, mask)
        probs_fake = self.dis(generated_samples)
        probs_real = self.dis(batch)

        loss_d = self.d_loss(probs_fake, probs_real)
        loss_d.backward()
        self.d_optimizer.step()

        return generated_samples, loss_d

    def train_step_generator(self, noise, mask, batch):

        self.num_updates += 0.5

        # Generator
        self.gen.train()
        self.dis.eval()
        self.g_optimizer.zero_grad()

        generated_samples = self.gen(noise, mask)
        probs_fake = self.dis(generated_samples)

        # or with detach?
        loss_g = self.g_loss(probs_fake)

        if self.is_fmatch:
            
            # get internal features from D(x) and D(G(z))
            fake_feats = self.dis.int_outputs[0]
            _ = self.dis(batch)
            real_feats = self.dis.int_outputs[0]

            loss_mse = (fake_feats - real_feats)**2
            loss_mse = loss_mse.mean()

            loss_g += loss_mse

        #loss_roi = roi_loss(masks, generated_samples, batch)
        #loss_roi = ((masks - batch)**2).mean()
        #(loss_g + loss_roi).backward()
        loss_g.backward()
        self.g_optimizer.step()

        return generated_samples, loss_g

    #def backward(self, loss):

    #    loss.backward()
    #    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
    #    self.optimizer.step()
