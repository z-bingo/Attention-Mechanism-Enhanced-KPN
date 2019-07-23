import os, sys
sys.path.append('../')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kpn_data_provider import sRGBGamma
from simulations.simulation_data_provider import Sim_Dataset

from KPN import KPN
from Att_Weight_KPN import Att_Weight_KPN
from Att_KPN import Att_KPN
from utils.training_util import load_checkpoint, calculate_psnr, calculate_ssim

from torchvision.transforms import transforms

import argparse

Gains = (
    (0.0, 0.0),
    (5*(10**-3), (10**-3)),
    (2*10**-2, 4.3*10**-3),
    (5*10**-2, 10**-2),
    (8*10**-2, 4.3*10**-2)
)

# val_dict = {
#         'kpn_5x5',
#         'kpn_7x7',
#         'mkpn',
#         'gkpn',
#         'channel_att',
#         'spatial_att',
#         'att_kpn',
#         'weight_kpn'
#     }

val_dict = {
        'kpn_5x5',
        'kpn_7x7',
        'mkpn',
        'weight_kpn'
    }

def prediction(model, gt, noisy, white_level):
    _, pred = model(noisy.view(1, -1, 512, 512), noisy[:, :8, ...], white_level)
    pred = pred.clamp(0.0, 1.0)
    pred = sRGBGamma(pred)
    pred = pred.cpu()

    psnr = calculate_psnr(pred, gt)
    ssim = calculate_ssim(pred, gt)
    # psnr.append(psnr_t)
    # ssim.append(ssim_t)
    # print(pred.size())
    return pred, psnr, ssim


parser = argparse.ArgumentParser()
parser.add_argument('--noise', default=0, type=int, help='Noise Level')
args = parser.parse_args()

Noise_Level = args.noise
color = False

data_set = Sim_Dataset(
    dataset_dir='/home/bingo/burst-denoise/dataset/Adobe5K',
    sigma_r=Gains[Noise_Level][0],
    sigma_s=Gains[Noise_Level][1],
    color=color,
    sim_file='/home/bingo/burst-denoise/dataset/sim.txt'
    # img_format='Real'
)

print('Read noise {}, shot noise {}.'.format(Gains[Noise_Level][0], Gains[Noise_Level][1]))

data_loader = DataLoader(
    dataset=data_set,
    batch_size=1,
    shuffle=True,
    num_workers=8
)

if 'kpn_5x5' in val_dict:
    kpn_5x5 = KPN(
        color=color,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=False,
        spatial_att=False
    ).cuda()
    kpn_5x5 = nn.DataParallel(kpn_5x5)
    state = load_checkpoint('../models/kpn_aug/checkpoint', best_or_latest='best')
    kpn_5x5.load_state_dict(state['state_dict'])
    print('KPN with 5x5-sized kernel is loaded for iteration {}!'.format(state['global_iter']))
    psnr_kpn_5x5, ssim_kpn_5x5 = [],[]

if 'kpn_7x7' in val_dict:
    kpn_7x7 = KPN(
        color=color,
        burst_length=8,
        blind_est=False,
        kernel_size=[7],
        sep_conv=False,
        channel_att=False,
        spatial_att=False
    ).cuda()
    kpn_7x7 = nn.DataParallel(kpn_7x7)
    state = load_checkpoint('../models/kpn_aug_7x7/checkpoint', best_or_latest='best')
    kpn_7x7.load_state_dict(state['state_dict'])
    print('KPN with 7x7-sized model is loaded from iteration {}!'.format(state['global_iter']))
    psnr_kpn_7x7, ssim_kpn_7x7 = [],[]

if 'mkpn' in val_dict:
    mkpn = KPN(
        color=color,
        burst_length=8,
        blind_est=False,
        kernel_size=[1, 3, 5, 7, 9],
        sep_conv=True,
        channel_att=False,
        spatial_att=False
    ).cuda()
    mkpn = nn.DataParallel(mkpn)
    state = load_checkpoint('../models/mkpn/checkpoint', best_or_latest='best')
    mkpn.load_state_dict(state['state_dict'])
    print('MKPN is loaded from iteration {}!'.format(state['global_iter']))
    psnr_mkpn, ssim_mkpn = [],[]

if 'gkpn' in val_dict:
    gkpn = Att_Weight_KPN(
        color=color,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=True,
        spatial_att=True
    ).cuda()
    gkpn = nn.DataParallel(gkpn)
    state = load_checkpoint('../models/att_weight_kpn/checkpoint', best_or_latest='best')
    gkpn.load_state_dict(state['state_dict'])
    print('GKPN is loaded from iteration {}!'.format(state['global_iter']))
    psnr_gkpn, ssim_gkpn = [],[]

if 'channel_att' in val_dict:
    channel_att = Att_KPN(
        color=color,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=True,
        spatial_att=False
    ).cuda()
    channel_att = nn.DataParallel(channel_att)
    state = load_checkpoint('../models/channel_att_kpn/checkpoint', best_or_latest='best')
    channel_att.load_state_dict(state['state_dict'])
    print('Channel attention is loaded from iteration {}!'.format(state['global_iter']))
    psnr_ckpn, ssim_ckpn = [], []

if 'spatial_att' in val_dict:
    spatial_att = Att_KPN(
        color=color,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=False,
        spatial_att=True
    ).cuda()
    spatial_att = nn.DataParallel(spatial_att)
    state = load_checkpoint('../models/spatial_att_kpn/checkpoint', best_or_latest='best')
    spatial_att.load_state_dict(state['state_dict'])
    print('Spatial attention is loaded from iteration {}!'.format(state['global_iter']))
    psnr_skpn, ssim_skpn = [], []

if 'att_kpn' in val_dict:
    att_kpn = Att_KPN(
        color=color,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=True,
        spatial_att=True
    ).cuda()
    att_kpn = nn.DataParallel(att_kpn)
    state = load_checkpoint('../models/att_kpn_color/checkpoint', best_or_latest='best')
    att_kpn.load_state_dict(state['state_dict'])
    print('Attentional KPN is loaded from iteration {}!'.format(state['global_iter']))
    psnr_akpn, ssim_akpn = [], []

if 'weight_kpn' in val_dict:
    wkpn = Att_Weight_KPN(
        color=color,
        burst_length=8,
        blind_est=False,
        kernel_size=[5],
        sep_conv=False,
        channel_att=False,
        spatial_att=False
    ).cuda()
    wkpn = nn.DataParallel(wkpn)
    state = load_checkpoint('../models/weight_kpn/checkpoint', best_or_latest='best')
    wkpn.load_state_dict(state['state_dict'])
    print('WKPN is loaded from iteration {}!'.format(state['global_iter']))
    psnr_wkpn, ssim_wkpn = [], []


if __name__ == '__main__':
    if not os.path.exists('./eval_images'):
        os.mkdir('./eval_images')
    files = os.listdir('./eval_images')
    for f in files:
        os.remove(os.path.join('./eval_images', f))

    print('{} simulated images'.format(len(data_set)))

    psnr_noisy, ssim_noisy = [], []
    trans_rgb = transforms.ToPILImage()
    with torch.no_grad():
        index = 0
        cnt = 0
        data_loader= iter(data_loader)
        while index < 100 and cnt < 110:
            cnt += 1
            try:
                (noisy, gt, white_level) = next(data_loader)
                index += 1
                noisy = noisy.cuda()
                gt = gt.cuda()
                white_level = white_level.cuda()

                gt = gt / white_level
                gt = gt.clamp(0.0, 1.0)
                gt = sRGBGamma(gt)

                if 'kpn_5x5' in val_dict:
                    _, kpn_5x5_pred = kpn_5x5(noisy.view(1, -1, 512, 512), noisy[:, :8, ...], white_level)
                    kpn_5x5_pred = kpn_5x5_pred.clamp(0.0, 1.0)
                    kpn_5x5_pred = sRGBGamma(kpn_5x5_pred)
                    kpn_5x5_pred = kpn_5x5_pred.cpu()
                    psnr_t = calculate_psnr(kpn_5x5_pred, gt)
                    ssim_t = calculate_ssim(kpn_5x5_pred, gt)
                    psnr_kpn_5x5.append(psnr_t)
                    ssim_kpn_5x5.append(ssim_t)
                    trans_rgb(kpn_5x5_pred.squeeze()).save('./eval_images/{}_kpn_5x5_{:.2f}dB_{:.4f}.png'.format(index, psnr_t, ssim_t), quality=100)
                    del kpn_5x5_pred

                if 'kpn_7x7' in val_dict:
                    _, kpn_7x7_pred = kpn_7x7(noisy.view(1, -1, 512, 512), noisy[:, :8, ...], white_level)
                    kpn_7x7_pred = kpn_7x7_pred.clamp(0.0, 1.0)
                    kpn_7x7_pred = sRGBGamma(kpn_7x7_pred)
                    kpn_7x7_pred = kpn_7x7_pred.cpu()
                    psnr_t = calculate_psnr(kpn_7x7_pred, gt)
                    ssim_t = calculate_ssim(kpn_7x7_pred, gt)
                    psnr_kpn_7x7.append(psnr_t)
                    ssim_kpn_7x7.append(ssim_t)
                    trans_rgb(kpn_7x7_pred.squeeze()).save('./eval_images/{}_kpn_7x7_{:.2f}dB_{:.4f}.png'.format(index, psnr_t, ssim_t), quality=100)
                    del kpn_7x7_pred

                if 'mkpn' in val_dict:
                    _, mkpn_pred = mkpn(noisy.view(1, -1, 512, 512), noisy[:, :8, ...], white_level)
                    mkpn_pred = mkpn_pred.clamp(0.0, 1.0)
                    mkpn_pred = sRGBGamma(mkpn_pred)
                    mkpn_pred = mkpn_pred.cpu()
                    psnr_t = calculate_psnr(mkpn_pred, gt)
                    ssim_t = calculate_ssim(mkpn_pred, gt)
                    psnr_mkpn.append(psnr_t)
                    ssim_mkpn.append(ssim_t)
                    trans_rgb(mkpn_pred.squeeze()).save('./eval_images/{}_mkpn_{:.2f}dB_{:.4f}.png'.format(index, psnr_t, ssim_t), quality=100)
                    del mkpn_pred

                if 'gkpn' in val_dict:
                    _, gkpn_pred = gkpn(noisy.view(1, -1, 512, 512), noisy[:, :8, ...], white_level)
                    gkpn_pred = gkpn_pred.clamp(0.0, 1.0)
                    gkpn_pred = sRGBGamma(gkpn_pred)
                    gkpn_pred = gkpn_pred.cpu()
                    psnr_t = calculate_psnr(gkpn_pred, gt)
                    ssim_t = calculate_ssim(gkpn_pred, gt)
                    psnr_gkpn.append(psnr_t)
                    ssim_gkpn.append(ssim_t)
                    trans_rgb(gkpn_pred.squeeze()).save('./eval_images/{}_gkpn_{:.2f}dB_{:.4f}.png'.format(index, psnr_t, ssim_t), quality=100)
                    del gkpn_pred

                if 'channel_att' in val_dict:
                    _, ckpn_pred = channel_att(noisy.view(1, -1, 512, 512), noisy[:, :8, ...], white_level)
                    ckpn_pred = ckpn_pred.clamp(0.0, 1.0)
                    ckpn_pred = sRGBGamma(ckpn_pred)
                    ckpn_pred = ckpn_pred.cpu()
                    psnr_t = calculate_psnr(ckpn_pred, gt)
                    ssim_t = calculate_ssim(ckpn_pred, gt)
                    psnr_ckpn.append(psnr_t)
                    ssim_ckpn.append(ssim_t)
                    del ckpn_pred

                if 'spatial_att' in val_dict:
                    _, skpn_pred = spatial_att(noisy.view(1, -1, 512, 512), noisy[:, :8, ...], white_level)
                    skpn_pred = skpn_pred.clamp(0.0, 1.0)
                    skpn_pred = sRGBGamma(skpn_pred)
                    skpn_pred = skpn_pred.cpu()
                    psnr_t = calculate_psnr(skpn_pred, gt)
                    ssim_t = calculate_ssim(skpn_pred, gt)
                    psnr_skpn.append(psnr_t)
                    ssim_skpn.append(ssim_t)
                    del skpn_pred

                if 'att_kpn' in val_dict:
                    _, akpn_pred = att_kpn(noisy.view(1, -1, 512, 512), noisy[:, :8, ...], white_level)
                    akpn_pred = akpn_pred.clamp(0.0, 1.0)
                    akpn_pred = sRGBGamma(akpn_pred)
                    akpn_pred = akpn_pred.cpu()
                    psnr_t = calculate_psnr(akpn_pred, gt)
                    ssim_t = calculate_ssim(akpn_pred, gt)
                    psnr_akpn.append(psnr_t)
                    ssim_akpn.append(ssim_t)
                    del akpn_pred

                if 'weight_kpn' in val_dict:
                    _, wkpn_pred = wkpn(noisy.view(1, -1, 512, 512), noisy[:, :8, ...], white_level)
                    wkpn_pred = wkpn_pred.clamp(0.0, 1.0)
                    wkpn_pred = sRGBGamma(wkpn_pred)
                    wkpn_pred = wkpn_pred.cpu()
                    psnr_t = calculate_psnr(wkpn_pred, gt)
                    ssim_t = calculate_ssim(wkpn_pred, gt)
                    psnr_wkpn.append(psnr_t)
                    ssim_wkpn.append(ssim_t)
                    trans_rgb(wkpn_pred.squeeze()).save('./eval_images/{}_wkpn_{:.2f}dB_{:.4f}.png'.format(index, psnr_t, ssim_t), quality=100)
                    del wkpn_pred

                noisy = sRGBGamma(noisy / white_level)
                noisy = noisy.cpu()
                psnr_t = calculate_psnr(noisy[:, 0, ...], gt)
                ssim_t = calculate_ssim(noisy[:, 0, ...], gt)
                psnr_noisy.append(psnr_t)
                ssim_noisy.append(ssim_t)
                trans_rgb(noisy[0, 0, ...]).save('./eval_images/{}_noisy_{:.2f}dB_{:.4f}.png'.format(index, psnr_t, ssim_t))

                gt = gt.cpu()
                trans_rgb(gt[0, ...]).save('./eval_images/{}_gt.png'.format(index))

                print('Image {} is OK!'.format(index))
            except:
                # print('Errors occur')
                pass

        print('Validation Over!')

        if 'kpn_5x5' in val_dict:
            print('KPN 5x5: PSNR {:.2f}dB, SSIM {:.4f}'.format(sum(psnr_kpn_5x5)/len(psnr_kpn_5x5), sum(ssim_kpn_5x5)/len(ssim_kpn_5x5)))

        if 'kpn_7x7' in val_dict:
            print('KPN 7x7: PSNR {:.2f}dB, SSIM {:.4f}'.format(sum(psnr_kpn_7x7)/len(psnr_kpn_7x7), sum(ssim_kpn_7x7)/len(ssim_kpn_7x7)))

        if 'mkpn' in val_dict:
            print('MKPN: PSNR {:.2f}dB, SSIM {:.4f}'.format(sum(psnr_mkpn)/len(psnr_mkpn), sum(ssim_mkpn)/len(ssim_mkpn)))

        if 'gkpn' in val_dict:
            print('GKPN: PSNR {:.2f}dB, SSIM {:.4f}'.format(sum(psnr_gkpn)/len(psnr_gkpn), sum(ssim_gkpn)/len(ssim_gkpn)))

        if 'channel_att' in val_dict:
            print('Channel attention KPN: PSNR {:.2f}dB, SSIM {:.4f}'.format(sum(psnr_ckpn)/len(psnr_ckpn), sum(ssim_ckpn)/len(ssim_ckpn)))

        if 'spatial_att' in val_dict:
            print('Spatial attention KPN: PSNR {:.2f}dB, SSIM {:.4f}'.format(sum(psnr_skpn)/len(psnr_skpn), sum(ssim_skpn)/len(ssim_skpn)))

        if 'att_kpn' in val_dict:
            print('Attentional KPN: PSNR {:.2f}dB, SSIM {:.4f}'.format(sum(psnr_akpn)/len(psnr_akpn), sum(ssim_akpn)/len(ssim_akpn)))

        if 'weight_kpn' in val_dict:
            print('Weight KPN: PSNR {:.2f}dB, SSIM {:.4f}'.format(sum(psnr_wkpn)/len(psnr_wkpn), sum(ssim_wkpn)/len(ssim_wkpn)))

        print('Noisy: PSNR {:.2f}dB, SSIM {:.4f}'.format(sum(psnr_noisy) / len(psnr_noisy), sum(ssim_noisy) / len(ssim_noisy)))

