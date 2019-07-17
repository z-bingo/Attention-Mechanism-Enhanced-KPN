import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kpn_data_provider import TrainDataSet
from utils.training_util import MovingAverage, save_checkpoint, load_checkpoint, read_config
import numpy as np
import torch.optim as optim
import argparse

class Network(nn.Module):
    def __init__(self, color=False):
        super(Network, self).__init__()
        channel = 3 if color else 1
        self.conv = nn.Sequential(
            nn.Conv2d(channel, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, channel, 3, 1, 1)
        )

    def forward(self, data):
        return self.conv(data)

class LossFn(nn.Module):
    def __init__(self, alpha=0.3, lambda_l=0.1):
        super(LossFn, self).__init__()
        self.alpha = alpha
        self.lambda_l = lambda_l

    def gradient(self, pred):
        h, w = pred.size()[-2:]
        t = torch.zeros_like(pred)
        t[..., 0:w-1] = t[..., 1:w]
        t1 = torch.sum((pred - t)**2)
        t = torch.zeros_like(pred)
        t[..., 0:h-1, :] = t[..., 1:h, :]
        t2 = torch.sum((pred-gt)**2)
        return t1+t2

    def forward(self, pred, gt):
        mask = torch.ones_like(pred)
        mask[pred <= gt] = 0
        Lasymm = torch.sum(torch.abs(self.alpha - mask) * ((pred - gt)**2))
        return Lasymm + self.lambda_l*self.gradient(pred)

parser = argparse.ArgumentParser()
parser.add_argument('--restart', '-r', action='store_true')
args = parser.parse_args()

config = read_config('kpn_specs/att_kpn_config.conf', 'kpn_specs/configspec.conf')
train_config = config['training']
data_set = TrainDataSet(
    train_config['dataset_configs'],
    img_format='.bmp',
    degamma=True,
    color=True,
    blind=False
)
data_loader = DataLoader(
    dataset=data_set,
    batch_size=32,
    shuffle=True,
    num_workers=16
)

loss_fn = LossFn()

model = Network(True).cuda()
model = nn.DataParallel(model)

model.train()

optimizer = optim.Adam(model.parameters(), lr=5e-5)

if not args.restart:
    state = load_checkpoint('./noise_models', best_or_latest='best')
    global_iter = state['global_iter']
    model.load_state_dict(state['model'])
    print('model load OK at iter {}'.format(global_iter))
else:
    global_iter = 0
min_loss = np.inf
loss_ave = MovingAverage(200)

import os
if not os.path.exists('./noise_models'):
    os.mkdir('./noise_models')

for epoch in range(500):
    for step, (data, A, B) in enumerate(data_loader):
        feed = data[:, 0, ...].cuda()
        gt = data[:, -1, ...].cuda()
        # print(data.size())
        pred = model(feed)

        loss = loss_fn(pred, gt)

        global_iter += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_ave.update(loss)
        if global_iter % 200 == 0:
            loss_t = loss_ave.get_value()
            min_loss = min(min_loss, loss_t)
            is_best = min_loss == loss_t
            save_checkpoint(
                {'model': model.state_dict(), 'global_iter': global_iter},
                is_best=is_best,
                checkpoint_dir='./noise_models',
                n_iter=global_iter
            )
        print('{: 6d}, epoch {: 3d}, iter {: 4d}, loss {:.4f}'.format(global_iter, epoch, step, loss))
