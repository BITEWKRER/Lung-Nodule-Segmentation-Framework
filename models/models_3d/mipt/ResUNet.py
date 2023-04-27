# -*-coding:utf-8 -*-
"""
# Time       ：2022/10/8 14:28
# Author     ：comi
# version    ：python 3.8
# Description：
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUNet(nn.Module):
    def __init__(self, training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            # nn.PReLU(16),
            nn.LeakyReLU(),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            # nn.PReLU(16),
            nn.LeakyReLU(),

            nn.Conv3d(32, 64, 2, 2),
            nn.InstanceNorm3d(64),
            # nn.PReLU(16),
            nn.LeakyReLU(),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            # nn.PReLU(32),
            nn.LeakyReLU(),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            # nn.PReLU(32),
            nn.LeakyReLU(),

            nn.Conv3d(64, 256, 2, 2),
            nn.InstanceNorm3d(256),
            # nn.PReLU(32),
            nn.LeakyReLU(),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            # nn.PReLU(64),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, 3, 1, 1),
            nn.InstanceNorm3d(256),
            # nn.PReLU(64),
            nn.LeakyReLU(),

            nn.Conv3d(256, 512, 2, 2),
            nn.InstanceNorm3d(512),
            # nn.PReLU(64),
            nn.LeakyReLU(),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.InstanceNorm3d(128),
            # nn.PReLU(128),
            nn.LeakyReLU(),

            nn.Conv3d(128, 128, 3, 1, 1),
            nn.InstanceNorm3d(128),
            # nn.PReLU(128),
            nn.LeakyReLU(),

            nn.Conv3d(128, 256, 2, 2),
            nn.InstanceNorm3d(256),
            # nn.PReLU(128),
            nn.LeakyReLU(),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            # nn.PReLU(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            # nn.PReLU(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            # nn.PReLU(256),
            nn.LeakyReLU(),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            # nn.PReLU(128),
            nn.LeakyReLU(),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            # nn.PReLU(128),
            nn.LeakyReLU(),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            # nn.PReLU(128),
            nn.LeakyReLU(),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            # nn.PReLU(64),
            nn.LeakyReLU(),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            # nn.PReLU(64),
            nn.LeakyReLU(),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            # nn.PReLU(64),
            nn.LeakyReLU(),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            # nn.PReLU(32),
            nn.LeakyReLU(),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            # nn.PReLU(32),
            nn.LeakyReLU(),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.InstanceNorm3d(32),
            # nn.PReLU(32)
            nn.LeakyReLU(),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.InstanceNorm3d(64),
            # nn.PReLU(64)
            nn.LeakyReLU(),
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.InstanceNorm3d(128),
            # nn.PReLU(128)
            nn.LeakyReLU(),
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            # nn.PReLU(256)
            nn.LeakyReLU(),
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.InstanceNorm3d(128),
            # nn.PReLU(128)
            nn.LeakyReLU(),
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.InstanceNorm3d(64),
            # nn.PReLU(64)
            nn.LeakyReLU(),
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.InstanceNorm3d(32),
            # nn.PReLU(32)
            nn.LeakyReLU(),
        )

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.dorp_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.dorp_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.dorp_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        return outputs