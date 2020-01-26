import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
from Att_KPN import Att_KPN
from KPN import KPN
from Att_Weight_KPN import Att_Weight_KPN
from noise_estimation import Network
from torchvision.transforms import transforms
import argparse
from utils.training_util import load_checkpoint
import os, sys
from PIL import Image
from utils.image_utils import center_crop_tensor
import visdom

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='./real_world/Nikon800_plant', help='A image or a directory')
    parser.add_argument('--read', default=0.2, type=float)
    parser.add_argument('--shot', default=0.1, type=float)
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
    gkpn_model = Att_Weight_KPN(
        color=False,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=True,
        spatial_att=True
    )
    gkpn_model = nn.DataParallel(gkpn_model.cuda())

    state = load_checkpoint('../models/att_weight_kpn/checkpoint', best_or_latest='best')
    gkpn_model.load_state_dict(state['state_dict'])
    gkpn_model.eval()

    wkpn_model = Att_Weight_KPN(
        color=False,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=False,
        spatial_att=False
    )
    wkpn_model = nn.DataParallel(wkpn_model.cuda())

    state = load_checkpoint('../models/weight_kpn/checkpoint', best_or_latest='best')
    wkpn_model.load_state_dict(state['state_dict'])
    wkpn_model.eval()

    kpn_model = KPN(
        color=False,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=False,
        spatial_att=False
    )
    kpn_model = nn.DataParallel(kpn_model.cuda())

    state = load_checkpoint('../models/kpn_aug/checkpoint', best_or_latest='best')
    kpn_model.load_state_dict(state['state_dict'])
    kpn_model.eval()

    noise_est = nn.DataParallel(Network(True).cuda())
    state = load_checkpoint('../noise_models', best_or_latest='best')
    noise_est.load_state_dict(state['model'])
    noise_est.eval()

    trans = transforms.ToTensor()
    imgs = []
    with torch.no_grad():
        if os.path.exists(args.img):
            if os.path.isdir(args.img):
                files = os.listdir(args.img)
                # file_index = np.random.permutation(len(files))[:8]
                file_index = range(8)
                for index in file_index:
                    img = Image.open(os.path.join(args.img, files[index]))
                    img = trans(img)

                    img += (0.02+0.01*img) * torch.randn_like(img)
                    imgs.append(img.clamp(0.0, 1.0))
            else:
                raise ValueError('should be a burst of frames, not a image!')
            s_read, s_shot = torch.Tensor([[[args.read]]]), torch.Tensor([[[args.shot]]])
            noise_est = torch.sqrt(s_read**2 + s_shot*torch.max(torch.zeros_like(imgs[0]), imgs[0]))
            imgs.append(noise_est)
            imgs = torch.stack(imgs, dim=0).unsqueeze(0).cuda()

            # imgs = torch.stack(imgs, dim=0).unsqueeze(0).cuda()
            # h, w = imgs.size()[-2:]
            # noise_est = noise_est(imgs[:, 0, ...].expand(1, 3, h, w))[:, 0, ...].unsqueeze(1).unsqueeze(2)
            # imgs = torch.cat([imgs, 15*noise_est], dim=1)

        else:
            raise ValueError('The path for image is not existing.')

        b, N, c, h, w = imgs.size()
        res_wkpn = torch.zeros(c, h, w).cuda()
        res_gkpn = torch.zeros(c, h, w).cuda()
        res_kpn = torch.zeros(c, h, w).cuda()

        res_gkpn_pred_i = torch.zeros(8, c, h, w).cuda()
        res_gkpn_residual = torch.zeros(8, c, h, w).cuda()

        patch_size = 512
        receptiveFiled = 120
        imgs_pad = torch.zeros(b, N, c, h + 2 * receptiveFiled, w + 2 * receptiveFiled)
        imgs_pad[..., receptiveFiled:-receptiveFiled, receptiveFiled:-receptiveFiled] = imgs

        if not os.path.exists('./eval_images_real'):
            os.mkdir('./eval_images_real')

        filename = os.path.basename(args.img)
        filename = os.path.splitext(filename)[0]
        trans = transforms.ToPILImage()

        for channel in range(c):
            for i in range(0, h, patch_size):
                for j in range(0, w, patch_size):
                    if i + patch_size <= h and j + patch_size <= w:
                        # feed = imgs[..., i:i+patch_size, j:j+patch_size].contiguous()
                        feed = imgs_pad[..., channel, i:i + patch_size + 2 * receptiveFiled,
                               j:j + patch_size + 2 * receptiveFiled].contiguous()
                    elif i + patch_size <= h:
                        # feed = imgs[..., i:i + patch_size, j:].contiguous()
                        feed = imgs_pad[..., channel, i:i + patch_size + 2 * receptiveFiled, j:].contiguous()
                    elif j + patch_size <= w:
                        # feed = imgs[..., i:, j:j+patch_size].contiguous()
                        feed = imgs_pad[..., channel, i:, j:j + patch_size + 2 * receptiveFiled].contiguous()
                    else:
                        # feed = imgs[..., i:, j:].contiguous()
                        feed = imgs_pad[..., channel, i:, j:].contiguous()

                    hs, ws = feed.size()[-2:]
                    hs -= 2*receptiveFiled
                    ws -= 2*receptiveFiled

                    feed = padding(feed, patch_size+2*receptiveFiled, patch_size+2*receptiveFiled)

                    # _, pred = wkpn_model(feed.view(b, -1, patch_size+2*receptiveFiled, patch_size+2*receptiveFiled), feed[:, 0:8, ...])
                    # res_wkpn[channel, i:i+patch_size, j:j+patch_size] = pred[..., receptiveFiled:hs+receptiveFiled, receptiveFiled:ws+receptiveFiled].squeeze()

                    pred_i, pred, residuals = gkpn_model(feed.view(b, -1, patch_size + 2 * receptiveFiled, patch_size + 2 * receptiveFiled), feed[:, 0:8, ...])
                    res_gkpn[channel, i:i + patch_size, j:j + patch_size] = pred[..., receptiveFiled:hs + receptiveFiled,
                                                                      receptiveFiled:ws + receptiveFiled].squeeze()
                    res_gkpn_pred_i[:, channel, i:i+patch_size, j:j+patch_size] = pred_i[..., receptiveFiled:hs + receptiveFiled,
                                                                      receptiveFiled:ws + receptiveFiled].squeeze()
                    res_gkpn_residual[:, channel, i:i + patch_size, j:j + patch_size] = residuals[...,
                                                                                      receptiveFiled:hs + receptiveFiled,
                                                                                      receptiveFiled:ws + receptiveFiled].squeeze()


                    # _, pred = kpn_model(feed.view(b, -1, patch_size + 2 * receptiveFiled, patch_size + 2 * receptiveFiled),
                    #                      feed[:, 0:8, ...])
                    # res_kpn[channel, i:i + patch_size, j:j + patch_size] = pred[..., receptiveFiled:hs + receptiveFiled,
                    #                                                   receptiveFiled:ws + receptiveFiled].squeeze()
                    print('{}, {} OK!'.format(i, j))

        res_kpn = res_kpn.cpu().clamp(0.0, 1.0)
        res_wkpn = res_wkpn.cpu().clamp(0.0, 1.0)
        res_gkpn = res_gkpn.cpu().clamp(0.0, 1.0)

        # trans(res_kpn).save('./eval_images_real/{}_pred_kpn.png'.format(filename), quality=100)
        # trans(res_wkpn).save('./eval_images_real/{}_pred_wkpn.png'.format(filename), quality=100)
        trans(res_gkpn).save('./eval_images_real/{}_pred_gkpn.png'.format(filename), quality=100)
        trans(imgs[0, 0, ...].cpu()).save('./eval_images_real/{}_noisy.png'.format(filename), quality=100)
        print('OK!')

if __name__ == '__main__':
    args = args_parser()
    validation(args)
