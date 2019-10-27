import torch
from torch.autograd import Variable
from Losses import roi_loss

class Trainer:
    def __init__(self, models, optimizers, losses, clip_norm,
        writer, num_updates, device, multi_gpu):

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

        self.criterion = criterion
        self.clip_norm = clip_norm
        self.writer = writer
        self.num_updates = num_updates

    def train_step(self, noise, masks, batch):

        self.gen.train()
        self.dis.train()
        self.num_updates += 1

        # Generator
        self.g_optimizer.zero_grad()
        generated_samples = self.gen(noise, masks)
        probs_generated = self.dis(generated_samples)

        loss_g = self.g_loss(probs_generated)
        #loss_roi = roi_loss(masks, generated_samples, batch)
        #loss_roi = ((masks - batch)**2).mean()
        #(loss_g + loss_roi).backward()
        loss_g.backward()
        self.g_optimizer.step()

        # Discriminator
        self.d_optimizer.zero_grad()
        probs_real = self.dis(batch)

        # with or without detach?
        loss_d = self.d_loss(probs_generated.detach(), probs_real)
        loss_d.backward()
        self.d_optimizer.step()

        return generated_samples, loss_g, loss_d

    def test_step(self, noise, masks, batch):

        self.gen.eval()
        self.dis.eval()

        generated_samples = self.gen(noise, masks)
        probs_fake = self.dis(generated_samples)
        probs_real = self.dis(batch)

        loss_g = self.g_loss(probs_generated)
        loss_d = self.d_loss(probs_generated, probs_real)

        return generated_samples, loss_g, loss_d

    #def backward(self, loss):

    #    loss.backward()
    #    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
    #    self.optimizer.step()
