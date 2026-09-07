"""
U-Net 2D 对比模型

与主模型的区别：
- 纯空间域卷积，无频域操作
- 频率信息展开成常数通道拼入输入（和 fno_baseline 相同处理）
- 经典编码器-解码器结构 + skip connection
- 无超网络、无注意力

用途：对比实验，证明频域操作对电磁场预测的必要性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.net(x)


class UNet2d(nn.Module):
    """
    4层 U-Net，参数量约 800 万，和主模型接近。
    接口与主模型完全兼容。
    """
    def __init__(self, modes1=12, modes2=12, width=48, n_layers=5,
                 in_channels=2, out_channels=6, freq_dim=32):
        super().__init__()
        # 输入：SDF + Mask + freq_channel = 3
        ch = [32, 64, 128, 256]

        # Encoder
        self.enc1 = ConvBlock(3,     ch[0])
        self.enc2 = ConvBlock(ch[0], ch[1])
        self.enc3 = ConvBlock(ch[1], ch[2])
        self.enc4 = ConvBlock(ch[2], ch[3])

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(ch[3], ch[3] * 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(ch[3]*2, ch[3], 2, stride=2)
        self.dec4 = ConvBlock(ch[3]*2, ch[3])
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
        self.dec3 = ConvBlock(ch[2]*2, ch[2])
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.dec2 = ConvBlock(ch[1]*2, ch[1])
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.dec1 = ConvBlock(ch[0]*2, ch[0])

        self.out_conv = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, x, freq=None):
        B, C, H, W = x.shape

        # 频率展开成常数通道
        if freq is not None:
            freq_map = freq.view(B, 1, 1, 1).expand(B, 1, H, W)
        else:
            freq_map = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        x = torch.cat([x, freq_map], dim=1)  # (B, 3, H, W)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # 输出尺寸对齐（U-Net 下采样可能导致尺寸不整除）
        out = self.out_conv(d1)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
