import torch
from torch.autograd import Variable

class Trainer:
    def __init__(self, model, optimizer, criterion, metric, clip_norm, writer, device):

        self.device = device

        if self.device.type == 'cuda':
            self.model = model.cuda()
        else:
            self.model = model

        self.optimizer = optimizer
        self.criterion = criterion
        self.clip_norm = clip_norm
        self.writer = writer
        self.num_updates = 0

    def train_step(self, batch):

        self.model.train()
        self.optimizer.zero_grad()
        self.num_updates += 1
        features, responses_real = batch
        responses_pred = self.model(features)
        loss = self.criterion(responses_pred, responses_real)
        self.backward(loss)
        quality = metric(responses_pred, responses_real)
        return loss, quality

    def test_step(self, batch):

        self.model.eval()
        features, responses_real = batch
        responses_pred = self.model(features)

        loss = self.criterion(responses_pred, responses_real)
        quality = metric(responses_pred, responses_real)
        return loss, quality

    def backward(self, loss):

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
