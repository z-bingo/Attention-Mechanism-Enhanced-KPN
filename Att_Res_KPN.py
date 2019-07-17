import torch
import torch.nn as nn
from KPN import *

class Att_Res_KPN(nn.Module):
    def __init__(self, color=True, burst_length=8, blind_est=False, kernel_size=[5], sep_conv=False,
                 channel_att=False, spatial_att=False, upMode='bilinear', core_bias=False):
        super(Att_Res_KPN, self).__init__()
        self.upMode = upMode
        self.burst_length = burst_length
        self.core_bias = core_bias
        self.color_channel = 3 if color else 1
        in_channel = (3 if color else 1) * (burst_length if blind_est else burst_length + 1)
        out_channel = (3 if color else 1) * (
            2 * sum(kernel_size) if sep_conv else np.sum(np.array(kernel_size) ** 2)) * burst_length
        if core_bias:
            out_channel += (3 if color else 1) * burst_length
        # 各个卷积层定义
        # 2~5层都是均值池化+3层卷积
        self.conv1 = Basic(in_channel, 64, channel_att=False, spatial_att=False)
        self.conv2 = Basic(64, 128, channel_att=False, spatial_att=False)
        self.conv3 = Basic(128, 256, channel_att=False, spatial_att=False)
        self.conv4 = Basic(256, 512, channel_att=False, spatial_att=False)
        self.conv5 = Basic(512, 512, channel_att=False, spatial_att=False)
        # 6~8层要先上采样再卷积
        self.conv6 = Basic(512 + 512, 512, channel_att=channel_att, spatial_att=spatial_att)
        self.conv7 = Basic(256 + 512, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv8 = Basic(256 + 128, 256, channel_att=channel_att, spatial_att=spatial_att)
        self.conv9 = Basic(256+64, out_channel, channel_att=channel_att, spatial_att=spatial_att)
        self.out_kernel = nn.Sequential(
            Basic(out_channel, out_channel),
            nn.Conv2d(out_channel, out_channel, 1, 1, 0)
        )
        # residual branch
        self.conv10 = Basic(256+64, 128, channel_att=channel_att, spatial_att=spatial_att)
        self.out_res = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            Basic(64, self.burst_length, g=1),
            nn.Conv2d(self.burst_length, self.burst_length, 1, 1, 0)
        )

        self.conv11 = Basic(256+64, 128, channel_att=channel_att, spatial_att=spatial_att)
        self.out_weight = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0),
            Basic(64, self.burst_length, g=1),
            nn.Conv2d(self.burst_length, self.burst_length, 1, 1, 0),
            # nn.Softmax(dim=1)  #softmax 效果较差
            nn.Sigmoid()
        )

        self.kernel_pred = KernelConv(kernel_size, sep_conv, self.core_bias)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)

    # 前向传播函数
    def forward(self, data_with_est, data, white_level=1.0):
        """
        forward and obtain pred image directly
        :param data_with_est: if not blind estimation, it is same as data
        :param data:
        :return: pred_img_i and img_pred
        """
        conv1 = self.conv1(data_with_est)
        conv2 = self.conv2(F.avg_pool2d(conv1, kernel_size=2, stride=2))
        conv3 = self.conv3(F.avg_pool2d(conv2, kernel_size=2, stride=2))
        conv4 = self.conv4(F.avg_pool2d(conv3, kernel_size=2, stride=2))
        conv5 = self.conv5(F.avg_pool2d(conv4, kernel_size=2, stride=2))
        # 开始上采样  同时要进行skip connection
        conv6 = self.conv6(torch.cat([conv4, F.interpolate(conv5, scale_factor=2, mode=self.upMode)], dim=1))
        conv7 = self.conv7(torch.cat([conv3, F.interpolate(conv6, scale_factor=2, mode=self.upMode)], dim=1))
        conv8 = self.conv8(torch.cat([conv2, F.interpolate(conv7, scale_factor=2, mode=self.upMode)], dim=1))
        conv9 = self.conv9(torch.cat([conv1, F.interpolate(conv8, scale_factor=2, mode=self.upMode)], dim=1))
        # return channel K*K*N
        core = self.out_kernel(conv9)

        # residual branch
        conv10 = self.conv10(torch.cat([conv1, F.interpolate(conv8, scale_factor=2, mode=self.upMode)], dim=1))
        residual = self.out_res(conv10)

        conv11 = self.conv11(torch.cat([conv1, F.interpolate(conv8, scale_factor=2, mode=self.upMode)], dim=1))
        weight = self.out_weight(conv11)

        pred_i, _ = self.kernel_pred(data, core, white_level)
        # only for gray images now, supporting for RGB could be programed later
        pred_i = weight * pred_i + (1-weight) * residual
        pred = torch.mean(pred_i, dim=1, keepdim=False)
        # pred = torch.sum(pred_i * weight, dim=1, keepdim=False)
        return pred_i, pred
