import sys
import torch
import torch.nn as nn
import numpy as np
from Att_KPN import Att_KPN
from torchvision.transforms import transforms
import argparse
from utils.training_util import load_checkpoint
import os, sys
from PIL import Image
from utils.image_utils import center_crop_tensor

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='./test', help='A image or a directory')
    parser.add_argument('--read', default=0.01, type=float)
    parser.add_argument('--shot', default=0.01, type=float)
    return parser.parse_args()

def padding(tensor, h, w):
    shape = tensor.size()
    hc, wc = shape[-2:]
    if hc == h and wc == w:
        return tensor
    shape = shape[:-2] + (h, w)
    t = torch.zeros(shape)
    t[..., :hc, :wc] = tensor
    return t

def validation(args):
    model = Att_KPN(
        color=True,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=True,
        spatial_att=True
    )
    model = nn.DataParallel(model.cuda())
    print(os.path.abspath('./models/att_kpn_color/checkpoint'))
    state = load_checkpoint('./models/att_kpn_color/checkpoint', best_or_latest='best')
    model.load_state_dict(state['state_dict'])
    model.eval()

    trans = transforms.ToTensor()
    imgs = []
    with torch.no_grad():
        if os.path.exists(args.img):
            if os.path.isdir(args.img):
                files = os.listdir(args.img)
                for f in files:
                    img = Image.open(os.path.join(args.img, f))
                    imgs.append(trans(img))
            else:
                raise ValueError('should be a burst of frames, not a image!')
            s_read, s_shot = torch.Tensor([[[args.read]]]), torch.Tensor([[[args.shot]]])
            noise_est = torch.sqrt(s_read**2 + s_shot*torch.max(torch.zeros_like(imgs[0]), imgs[0]))
            imgs.append(noise_est)
            imgs = torch.stack(imgs, dim=0).unsqueeze(0)
            imgs = imgs.cuda()
        else:
            raise ValueError('The path for image is not existing.')

        b, N, c, h, w = imgs.size()
        res = torch.zeros(c, h, w).cuda()

        patch_size = 512
        receptiveFiled = 120
        imgs_pad = torch.zeros(b, N, c, h + 2 * receptiveFiled, w + 2 * receptiveFiled)
        imgs_pad[..., receptiveFiled:-receptiveFiled, receptiveFiled:-receptiveFiled] = imgs

        if not os.path.exists('./eval_images_real'):
            os.mkdir('./eval_images_real')

        filename = os.path.basename(args.img)
        filename = os.path.splitext(filename)[0]
        trans = transforms.ToPILImage()

        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                if i + patch_size <= h and j + patch_size <= w:
                    # feed = imgs[..., i:i+patch_size, j:j+patch_size].contiguous()
                    feed = imgs_pad[..., i:i + patch_size + 2 * receptiveFiled,
                           j:j + patch_size + 2 * receptiveFiled].contiguous()
                elif i + patch_size <= h:
                    # feed = imgs[..., i:i + patch_size, j:].contiguous()
                    feed = imgs_pad[..., i:i + patch_size + 2 * receptiveFiled, j:].contiguous()
                elif j + patch_size <= w:
                    # feed = imgs[..., i:, j:j+patch_size].contiguous()
                    feed = imgs_pad[..., i:, j:j + patch_size + 2 * receptiveFiled].contiguous()
                else:
                    # feed = imgs[..., i:, j:].contiguous()
                    feed = imgs_pad[..., i:, j:].contiguous()

                hs, ws = feed.size()[-2:]
                hs -= 2*receptiveFiled
                ws -= 2*receptiveFiled

                feed = padding(feed, patch_size+2*receptiveFiled, patch_size+2*receptiveFiled)

                _, pred = model(feed.view(b, -1, patch_size+2*receptiveFiled, patch_size+2*receptiveFiled), feed[:, 0:8, ...])
                res[:, i:i+patch_size, j:j+patch_size] = pred[..., receptiveFiled:hs+receptiveFiled, receptiveFiled:ws+receptiveFiled].squeeze()
                print('{}, {} OK!'.format(i, j))
        res = res.cpu().clamp(0.0, 1.0)

        trans(res).save('./eval_images_real/{}_pred.png'.format(filename), quality=100)
        trans(imgs[0, 0, ...].cpu()).save('./eval_images_real/{}_noisy.png'.format(filename), quality=100)
        print('OK!')

if __name__ == '__main__':
    args = args_parser()
    validation(args)
