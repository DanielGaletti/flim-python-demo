"""
U-Net: encoder VGG16 pretrained + decoder com skip connections
Arquitetura clássica adaptada para SOD (saída saliency map 1ch)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBnRelu(nn.Sequential):
    def __init__(self, cin, cout, k=3, p=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=p, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu(in_c + skip_c, out_c),
            ConvBnRelu(out_c, out_c),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained).features

        # Encoder (VGG16 blocks)
        self.enc1 = vgg[0:5]    # 64ch  /2
        self.enc2 = vgg[5:10]   # 128ch /4
        self.enc3 = vgg[10:17]  # 256ch /8
        self.enc4 = vgg[17:24]  # 512ch /16
        self.enc5 = vgg[24:31]  # 512ch /32

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBnRelu(512, 512),
            ConvBnRelu(512, 512),
        )

        # Decoder (U-Net)
        self.dec4 = DecoderBlock(512, 512, 256)
        self.dec3 = DecoderBlock(256, 256, 128)
        self.dec2 = DecoderBlock(128, 128,  64)
        self.dec1 = DecoderBlock( 64,  64,  32)

        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        e1 = self.enc1(x)   # /2
        e2 = self.enc2(e1)  # /4
        e3 = self.enc3(e2)  # /8
        e4 = self.enc4(e3)  # /16
        e5 = self.enc5(e4)  # /32

        b  = self.bottleneck(e5)

        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return F.interpolate(self.head(d1), (H, W), mode='bilinear', align_corners=False)
