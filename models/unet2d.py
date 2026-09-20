"""U-Net 2D 对比模型。

与主模型的区别：
- 纯空间域卷积，无频域操作
- 条件信息展开成常数通道拼入输入（和 fno_baseline 相同处理）
- 经典编码器-解码器结构 + skip connection
- 无超网络、无注意力

用途：对比实验，证明频域操作对电磁场预测的必要性。

注意：U-Net 不是 FNO，不继承 FNOBase。但它接受同一个 `cfg: ModelConfig`
（忽略 modes1/modes2/width/n_layers 等 FNO 专属字段）并保持 `forward(x, freq)` 签名。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import normalize_freq
from core.config import ModelConfig
from core.registry import register


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

    def forward(self, x):
        return self.net(x)


@register("unet")
class UNet2d(nn.Module):
    """4 层 U-Net，参数量与主模型量级相当。"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.in_channels
        self.cond_channels = int(cfg.cond_channels)
        ch = [32, 64, 128, 256]

        # Encoder：输入 = SDF + Mask + cond_channels
        self.enc1 = ConvBlock(self.in_channels + self.cond_channels, ch[0])
        self.enc2 = ConvBlock(ch[0], ch[1])
        self.enc3 = ConvBlock(ch[1], ch[2])
        self.enc4 = ConvBlock(ch[2], ch[3])

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(ch[3], ch[3] * 2)

        self.up4 = nn.ConvTranspose2d(ch[3] * 2, ch[3], 2, stride=2)
        self.dec4 = ConvBlock(ch[3] * 2, ch[3])
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
        self.dec3 = ConvBlock(ch[2] * 2, ch[2])
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.dec2 = ConvBlock(ch[1] * 2, ch[1])
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.dec1 = ConvBlock(ch[0] * 2, ch[0])

        self.out_conv = nn.Conv2d(ch[0], cfg.out_channels, 1)

    def _broadcast_condition(self, x, freq):
        B, _, H, W = x.shape
        if freq is None:
            return torch.zeros(B, self.cond_channels, H, W, device=x.device, dtype=x.dtype)
        # freq 已由 normalize_freq 规范成 (B, 2)；此处只做通道数与空间广播
        n_cond = freq.shape[1]
        if n_cond < self.cond_channels:
            pad = torch.zeros(B, self.cond_channels - n_cond, device=x.device, dtype=x.dtype)
            freq = torch.cat([freq, pad], dim=1)
        cond = freq[:, :self.cond_channels]
        return cond.view(B, self.cond_channels, 1, 1).expand(B, self.cond_channels, H, W)

    def forward(self, x, freq=None):
        B, C, H, W = x.shape
        freq = normalize_freq(freq, B)
        cond_map = self._broadcast_condition(x, freq)
        x = torch.cat([x, cond_map], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out_conv(d1)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out
