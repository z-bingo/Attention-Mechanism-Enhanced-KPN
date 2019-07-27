from __future__ import absolute_import

import sys
import os
sys.path.append('../')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.training_util import calculate_psnr, calculate_ssim, load_checkpoint

from eval_counterpart.eval_data_loader import TrainDataSet

from Att_KPN import Att_KPN
from KPN import KPN
# from mkpn import mkpn
from Att_Weight_KPN import Att_Weight_KPN

from kpn_data_provider import sRGBGamma
from torchvision.transforms import transforms
from PIL import Image
import sys

if __name__ == '__main__':
    dataset = TrainDataSet(
        dataset_dir='/home/bingo/burst-denoise/dataset/Adobe5K',
        burst_size=8,
        patch_size=512,
        upscale=1,
        img_format='.bmp',
        sigma_read=None,
        sigma_shot=None,
        color=False,
        blind=False
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4
    )

    att_kpn = Att_KPN(
        color=False,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=True,
        spatial_att=True,
        upMode='bilinear',
        core_bias=False
    )

    kpn = KPN(
        color=False,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=False,
        spatial_att=False,
        upMode='bilinear',
        core_bias=False
    )

    mkpn = KPN(
        color=False,
        burst_length=8,
        blind_est=False,
        kernel_size=[1,3,5,7,9],
        sep_conv=True,
        channel_att=False,
        spatial_att=False,
        upMode='bilinear',
        core_bias=False
    )

    att_weight_kpn = Att_Weight_KPN(
        color=False,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=True,
        spatial_att=True,
        upMode='bilinear',
        core_bias=False
    )

    att_kpn = nn.DataParallel(att_kpn.cuda())
    kpn = nn.DataParallel(kpn.cuda())
    mkpn = nn.DataParallel(mkpn.cuda())
    att_weight_kpn = nn.DataParallel(att_weight_kpn.cuda())

    # load
    # att_kpn_ckpt = load_checkpoint('../models/att_kpn/checkpoint', best_or_latest='best')
    kpn_ckpt = load_checkpoint('../models/kpn_aug/checkpoint', best_or_latest='best')
    mkpn_ckpt = load_checkpoint('../models/mkpn/checkpoint', best_or_latest='best')
    att_weight_kpn_ckpt = load_checkpoint('../models/att_weight_kpn/checkpoint', best_or_latest='best')

    # att_kpn.load_state_dict(att_kpn_ckpt['state_dict'])
    kpn.load_state_dict(kpn_ckpt['state_dict'])
    mkpn.load_state_dict(mkpn_ckpt['state_dict'])
    att_weight_kpn.load_state_dict(att_weight_kpn_ckpt['state_dict'])
    print('The models are load OK!')

    if not os.path.exists('./eval_images'):
        os.mkdir('./eval_images')
    else:
        files = os.listdir('./eval_images')
        for f in files:
            os.remove(os.path.join('./eval_images', f))

    trans = transforms.ToPILImage()

    psnr_kpn, psnr_att_kpn, psnr_mkpn, psnr_att_weight_kpn = [], [], [], []
    ssim_kpn, ssim_att_kpn, ssim_mkpn, ssim_att_weight_kpn = [], [], [], []

    with torch.no_grad():
        for i, (burst_noise, gt, white_level) in enumerate(data_loader):
            if i > 10:
                break

            burst_noise = burst_noise.cuda()
            gt = gt.cuda()
            white_level = white_level.cuda()
            # att_kpn
            # _, att_pred = att_kpn(burst_noise, burst_noise[:, 0:8, ...], white_level)
            # kpn
            _, pred = kpn(burst_noise, burst_noise[:, 0:8, ...], white_level)
            pred = pred.clamp(0.0, 1.0)
            # att_res
            _, mkpn_pred = mkpn(burst_noise, burst_noise[:, :8, ...], white_level)
            mkpn_pred = mkpn_pred.clamp(0.0, 1.0)
            #
            _, att_weight_pred = att_weight_kpn(burst_noise, burst_noise[:, :8, ...], white_level)
            att_weight_pred = att_weight_pred.clamp(0.0, 1.0)

            # sRGB Gamma
            # att_pred = sRGBGamma(att_pred)
            mkpn_pred = sRGBGamma(mkpn_pred)
            att_weight_pred = sRGBGamma(att_weight_pred)
            pred = sRGBGamma(pred)
            gt = sRGBGamma(gt)
            burst_noise = sRGBGamma(burst_noise/white_level)

            # clamp
            # att_pred = torch.clamp(att_pred, 0.0, 1.0)
            # pred = torch.clamp(pred, 0.0, 1.0)

            pred = pred.cpu()
            # att_pred = att_pred.cpu()
            mkpn_pred = mkpn_pred.cpu()
            att_weight_pred = att_weight_pred.cpu()
            gt = gt.cpu()
            burst_noise = burst_noise.cpu()
            # res = res.cpu()

            # save
            # psnr = calculate_psnr(att_pred.unsqueeze(1), gt.unsqueeze(1))
            # ssim = calculate_ssim(att_pred.unsqueeze(1), gt.unsqueeze(1))
            # trans(att_pred).save('./eval_images/{}_attKPN_{:.2f}dB_{:.4f}.png'.format(i, psnr, ssim), quality=100)
            # psnr_att_kpn.append(psnr)
            # ssim_att_kpn.append(ssim)

            # kpn
            psnr = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
            ssim = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
            trans(pred).save('./eval_images/{}_KPN_{:.2f}dB_{:.4f}.png'.format(i, psnr, ssim), quality=100)
            psnr_kpn.append(psnr)
            ssim_kpn.append(ssim)

            # att res kpn
            psnr = calculate_psnr(mkpn_pred.unsqueeze(1), gt.unsqueeze(1))
            ssim = calculate_ssim(mkpn_pred.unsqueeze(1), gt.unsqueeze(1))
            trans(mkpn_pred).save('./eval_images/{}_MKPN_{:.2f}dB_{:.4f}.png'.format(i, psnr, ssim), quality=100)
            psnr_mkpn.append(psnr)
            ssim_mkpn.append(ssim)

            # att weight kpn
            psnr = calculate_psnr(att_weight_pred.unsqueeze(1), gt.unsqueeze(1))
            ssim = calculate_ssim(att_weight_pred.unsqueeze(1), gt.unsqueeze(1))
            trans(att_weight_pred).save('./eval_images/{}_attWeightKPN_{:.2f}dB_{:.4f}.png'.format(i, psnr, ssim),
                                     quality=100)
            psnr_att_weight_kpn.append(psnr)
            ssim_att_weight_kpn.append(ssim)


            psnr = calculate_psnr(burst_noise[:, 0, ...].unsqueeze(1), gt.unsqueeze(1))
            ssim = calculate_ssim(burst_noise[:, 0, ...].unsqueeze(1), gt.unsqueeze(1))
            # for index in range(9):
            #     if index < 8:
            #         trans(burst_noise[0, index, ...]).save('./eval_images/{}_noisy_{:.2f}dB_{:.4f}_{}.png'.
            #                                                format(i, psnr, ssim, index), quality=100)
            #     else:
            #         trans(burst_noise[0, index, ...]).convert('P').save('./eval_images/{}_noisy_{:.2f}dB_{:.4f}_{}.png'.
            #                                                format(i, psnr, ssim, index), quality=100)

            trans(gt).save('./eval_images/{}_gt.png'.format(i), quality=100)

            print('Image {} is OK!'.format(i))
        print('KPN average PSNR {:.2f}dB'.format(sum(psnr_kpn)/len(psnr_kpn)))
        print('KPN average SSIM {:.4f}'.format(sum(ssim_kpn)/len(ssim_kpn)))
        # print('Attentional KPN average PSNR {:.2f}dB'.format(sum(psnr_att_kpn)/len(psnr_att_kpn)))
        # print('Attentional KPN average SSIM {:.4f}'.format(sum(ssim_att_kpn)/len(ssim_att_kpn)))
        print('Multi KPN average PSNR {:.2f}dB'.format(sum(psnr_mkpn) / len(psnr_mkpn)))
        print('Multi KPN average SSIM {:.4f}'.format(sum(ssim_mkpn) / len(ssim_mkpn)))
        print('Attentional Weighted KPN average PSNR {:.2f}dB'.format(sum(psnr_att_weight_kpn) / len(psnr_att_weight_kpn)))
        print('Attentional Weighted KPN average SSIM {:.4f}'.format(sum(ssim_att_weight_kpn) / len(ssim_att_weight_kpn)))