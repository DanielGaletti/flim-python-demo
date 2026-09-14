"""
MEANet: Multi-scale Edge-embedded Attention Network
Reimplementação baseada em: Liang et al., ESWA 2023
Backbone: VGG16 (ImageNet pretrained)
~3.27M params no decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBnRelu(nn.Sequential):
    def __init__(self, cin, cout, k=3, p=1, d=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class MEAModule(nn.Module):
    """Multi-scale Edge-embedded Attention"""
    def __init__(self, c):
        super().__init__()
        # Edge branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        # Multi-scale dilated convs: padding=dilation preserva spatial size
        self.d1 = ConvBnRelu(c, c // 4, k=3, p=1, d=1)
        self.d3 = ConvBnRelu(c, c // 4, k=3, p=3, d=3)
        self.d5 = ConvBnRelu(c, c // 4, k=3, p=5, d=5)
        self.d7 = ConvBnRelu(c, c // 4, k=3, p=7, d=7)
        self.fuse_ms = ConvBnRelu(c, c)

        # Spatial attention gate
        self.attn = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())
        self.out  = ConvBnRelu(c, c)

    def forward(self, x):
        edge = torch.sigmoid(self.edge_conv(x))
        ms   = self.fuse_ms(torch.cat(
            [self.d1(x), self.d3(x), self.d5(x), self.d7(x)], dim=1))
        attn = self.attn(ms)
        return self.out(x + ms * attn * (1 + edge))


class MSGModule(nn.Module):
    """Multi-level Semantic Guidance"""
    def __init__(self, high_c, low_c, out_c):
        super().__init__()
        self.compress_high = ConvBnRelu(high_c, out_c, k=1, p=0)
        self.compress_low  = ConvBnRelu(low_c,  out_c, k=1, p=0)
        self.gate = nn.Sequential(nn.Conv2d(out_c, out_c, 1), nn.Sigmoid())
        self.conv = ConvBnRelu(out_c, out_c)

    def forward(self, high, low):
        h = F.interpolate(self.compress_high(high), size=low.shape[2:],
                          mode='bilinear', align_corners=False)
        l = self.compress_low(low)
        g = self.gate(h)
        return self.conv(l * g + l)


class MEANet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = models.vgg16(pretrained=pretrained).features

        self.enc1 = vgg[0:5]    # 64ch, /2
        self.enc2 = vgg[5:10]   # 128ch, /4
        self.enc3 = vgg[10:17]  # 256ch, /8
        self.enc4 = vgg[17:24]  # 512ch, /16
        self.enc5 = vgg[24:31]  # 512ch, /32

        C = 64
        self.lat5 = ConvBnRelu(512, C, k=1, p=0)
        self.lat4 = ConvBnRelu(512, C, k=1, p=0)
        self.lat3 = ConvBnRelu(256, C, k=1, p=0)
        self.lat2 = ConvBnRelu(128, C, k=1, p=0)
        self.lat1 = ConvBnRelu(64,  C, k=1, p=0)

        self.mea5 = MEAModule(C)
        self.mea4 = MEAModule(C)
        self.mea3 = MEAModule(C)

        self.msg54 = MSGModule(C, C, C)
        self.msg43 = MSGModule(C, C, C)
        self.msg32 = MSGModule(C, C, C)
        self.msg21 = MSGModule(C, C, C)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.head  = nn.Conv2d(C, 1, 1)
        self.head5 = nn.Conv2d(C, 1, 1)
        self.head4 = nn.Conv2d(C, 1, 1)
        self.head3 = nn.Conv2d(C, 1, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        f5 = self.mea5(self.lat5(e5))
        f4 = self.mea4(self.lat4(e4))
        f3 = self.mea3(self.lat3(e3))
        f2 = self.lat2(e2)
        f1 = self.lat1(e1)

        d4 = self.msg54(f5, f4)
        d3 = self.msg43(d4, f3)
        d2 = self.msg32(d3, f2)
        d1 = self.msg21(d2, f1)

        out = F.interpolate(self.head(d1), (H, W), mode='bilinear', align_corners=False)
        if self.training:
            s5 = F.interpolate(self.head5(f5), (H, W), mode='bilinear', align_corners=False)
            s4 = F.interpolate(self.head4(d4), (H, W), mode='bilinear', align_corners=False)
            s3 = F.interpolate(self.head3(d3), (H, W), mode='bilinear', align_corners=False)
            return out, s3, s4, s5
        return out
