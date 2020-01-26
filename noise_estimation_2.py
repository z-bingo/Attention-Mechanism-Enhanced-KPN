import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kpn_data_provider import TrainDataSet
from utils.training_util import MovingAverage, save_checkpoint, load_checkpoint, read_config
import numpy as np
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter

class Network(nn.Module):
    def __init__(self, color=False):
        super(Network, self).__init__()
        channel = 3 if color else 1
        self.conv = nn.Sequential(
            nn.Conv2d(channel, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
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

def train():
    log_writer = SummaryWriter('./logs_2')
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
        num_workers=4
    )

    loss_fn = nn.L1Loss()

    model = Network(True).cuda()

    model.train()

    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    if not args.restart:
        model.load_state_dict(load_checkpoint('./noise_models_2', best_or_latest='best'))
    global_iter = 0
    min_loss = np.inf
    loss_ave = MovingAverage(200)

    import os
    if not os.path.exists('./noise_models_2'):
        os.mkdir('./noise_models_2')

    for epoch in range(100):
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

            log_writer.add_scalar('loss', loss, global_iter)

            loss_ave.update(loss)
            if global_iter % 200 == 0:
                loss_t = loss_ave.get_value()
                min_loss = min(min_loss, loss_t)
                is_best = min_loss == loss_t
                save_checkpoint(
                    model.state_dict(),
                    is_best=is_best,
                    checkpoint_dir='./noise_models_2',
                    n_iter=global_iter
                )
            print('{: 6d}, epoch {: 3d}, iter {: 4d}, loss {:.4f}'.format(global_iter, epoch, step, loss))

def eval():
    model = Network(True).cuda()
    model.load_state_dict(load_checkpoint('./noise_models_2', best_or_latest='best'))
    model.eval()
    from torchvision.transforms import transforms
    from PIL import Image
    img = Image.open('./003/1.jpg')
    trans_tensor = transforms.ToTensor()
    trans_srgb = transforms.ToPILImage()
    img = trans_tensor(img).unsqueeze(0).cuda()
    pred = model(img).squeeze()
    print('min:', torch.min(pred), 'max:', torch.max(pred))
    pred = pred / torch.max(pred)
    pred = pred.cpu()
    trans_srgb(pred).save('./003/1_pred.png', quality=100)
    print('OK!')


if __name__ == '__main__':
    eval()
    # train()
