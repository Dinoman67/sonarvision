"""
Custom modules for SSS (Side-Scan Sonar) optimized YOLOv8 variants.

Based on:
- SS-YOLO: Fast-C2f + GhostConv for lightweight SSS detection
- YOLOv8-ESI: SE attention for frequency-domain feature extraction

Author: Buffy (Codebuff)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletConv(nn.Module):
    """Wavelet Convolution for YOLOv8-ESI. Decomposes features into frequency sub-bands."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.register_buffer('haar_low', torch.tensor([1.0, 1.0]).view(1, 1, 2) / math.sqrt(2))
        self.register_buffer('haar_high', torch.tensor([1.0, -1.0]).view(1, 1, 2) / math.sqrt(2))

        self.conv_ll = nn.Conv2d(in_channels, out_channels // 4, kernel_size, stride, padding, bias=False)
        self.conv_lh = nn.Conv2d(in_channels, out_channels // 4, kernel_size, stride, padding, bias=False)
        self.conv_hl = nn.Conv2d(in_channels, out_channels // 4, kernel_size, stride, padding, bias=False)
        self.conv_hh = nn.Conv2d(in_channels, out_channels // 4, kernel_size, stride, padding, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def wavelet_2d(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0))

        x_low = F.conv1d(x.reshape(-1, 1, x.size(3)), self.haar_low, padding=1)[:, :, :x.size(3) // 2]
        x_high = F.conv1d(x.reshape(-1, 1, x.size(3)), self.haar_high, padding=1)[:, :, :x.size(3) // 2]
        x_low = x_low.reshape(B, C, H, -1)
        x_high = x_high.reshape(B, C, H, -1)

        ll = F.conv1d(x_low.reshape(-1, 1, H), self.haar_low, padding=1)[:, :, :H // 2].reshape(B, C, -1, x_low.size(3))
        lh = F.conv1d(x_low.reshape(-1, 1, H), self.haar_high, padding=1)[:, :, :H // 2].reshape(B, C, -1, x_low.size(3))
        hl = F.conv1d(x_high.reshape(-1, 1, H), self.haar_low, padding=1)[:, :, :H // 2].reshape(B, C, -1, x_high.size(3))
        hh = F.conv1d(x_high.reshape(-1, 1, H), self.haar_high, padding=1)[:, :, :H // 2].reshape(B, C, -1, x_high.size(3))
        return ll, lh, hl, hh

    def forward(self, x):
        ll, lh, hl, hh = self.wavelet_2d(x)
        out = torch.cat([self.conv_ll(ll), self.conv_lh(lh), self.conv_hl(hl), self.conv_hh(hh)], dim=1)
        return self.act(self.bn(out))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for parameter efficiency."""

    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        residual = x
        x = self.act(self.bn_dw(self.dw(x)))
        x = self.act(self.bn_pw(self.pw(x)))
        return x + residual if self.shortcut else x


class FastC2f(nn.Module):
    """Fast C2f Module for SS-YOLO. Uses depthwise separable convolutions."""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.cv2 = nn.Conv2d((n + 2) * hidden_channels // 2, out_channels, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bottlenecks = nn.ModuleList([
            DepthwiseSeparableConv(hidden_channels // 2, hidden_channels // 2, shortcut)
            for _ in range(n)
        ])
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.cv1(x)))
        x1, x2 = x.chunk(2, dim=1)
        y = [x1, x2]
        for bottleneck in self.bottlenecks:
            y.append(bottleneck(y[-1]))
        return self.act(self.bn2(self.cv2(torch.cat(y, dim=1))))


class GhostConv(nn.Module):
    """Ghost Convolution for SS-YOLO. Generates features via cheap operations."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3):
        super().__init__()
        init_channels = out_channels // ratio
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True)
        )
        self.cheap_op = nn.Sequential(
            nn.Conv2d(init_channels, init_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True)
        )
        self.n_new = out_channels - init_channels

    def forward(self, x):
        primary = self.primary_conv(x)
        cheap = self.cheap_op(primary)
        return torch.cat([primary, cheap], dim=1)[:, :primary.size(1) + self.n_new, :, :]


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.spatial_gate(torch.cat([avg_out, max_out], dim=1))
        b, c, _, _ = x.size()
        return x * self.channel_gate(x).view(b, c, 1, 1)
