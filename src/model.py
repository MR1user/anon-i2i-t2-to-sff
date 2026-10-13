"""U-Net used for paired T2w -> SFF (signal fat fraction) image-to-image translation."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Input : (B, 1, H, W) min-max normalised T2w slice in [0, 1]
    Output: (B, 1, H, W) SFF map in [0, 1] (sigmoid output)
    H and W must be multiples of 16 (256 x 256 recommended).
    """

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b = self.enc4(self.pool(s3))
        x = self.dec3(torch.cat([self.up(b), s3], dim=1))
        x = self.dec2(torch.cat([self.up(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up(x), s1], dim=1))
        return self.act(self.final(x))
