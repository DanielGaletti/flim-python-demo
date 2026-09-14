"""
MSCNet: Multi-Scale Context Network for Salient Object Detection
Reimplementação baseada em: Lin et al., ICPR 2022
Backbone: ResNet50 (ImageNet pretrained)
~3.26M params no decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBnRelu(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=1, dilation=1, relu=True):
        layers = [
            nn.Conv2d(cin, cout, k, s, p, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class MSCEModule(nn.Module):
    """Multi-Scale Context Extraction module"""
    def __init__(self, in_c, out_c):
        super().__init__()
        mid = out_c // 4
        self.d1  = ConvBnRelu(in_c, mid, k=3, p=1,  dilation=1)
        self.d2  = ConvBnRelu(in_c, mid, k=3, p=2,  dilation=2)
        self.d4  = ConvBnRelu(in_c, mid, k=3, p=4,  dilation=4)
        self.d8  = ConvBnRelu(in_c, mid, k=3, p=8,  dilation=8)
        self.fuse = ConvBnRelu(out_c, out_c, k=1, p=0)

    def forward(self, x):
        return self.fuse(torch.cat([self.d1(x), self.d2(x), self.d4(x), self.d8(x)], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(c, c // r), nn.ReLU(), nn.Linear(c // r, c), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.fc(self.gap(x).view(b, c))
        return x * w.view(b, c, 1, 1)


class APFAModule(nn.Module):
    """Attention-based Pyramid Feature Aggregation"""
    def __init__(self, c):
        super().__init__()
        self.ca   = ChannelAttention(c)
        self.conv = ConvBnRelu(c, c)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        fused = self.ca(x + skip)
        return self.conv(fused)


class MSCNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        res = models.resnet50(pretrained=pretrained)

        self.enc1 = nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool)  # 64ch, /4
        self.enc2 = res.layer1  # 256ch, /4
        self.enc3 = res.layer2  # 512ch, /8
        self.enc4 = res.layer3  # 1024ch, /16
        self.enc5 = res.layer4  # 2048ch, /32

        C = 64
        self.lat5 = ConvBnRelu(2048, C, k=1, p=0)
        self.lat4 = ConvBnRelu(1024, C, k=1, p=0)
        self.lat3 = ConvBnRelu(512,  C, k=1, p=0)
        self.lat2 = ConvBnRelu(256,  C, k=1, p=0)

        self.msce5 = MSCEModule(C, C)
        self.msce4 = MSCEModule(C, C)
        self.msce3 = MSCEModule(C, C)

        self.apfa4 = APFAModule(C)
        self.apfa3 = APFAModule(C)
        self.apfa2 = APFAModule(C)

        self.head = nn.Conv2d(C, 1, 1)
        self.head4 = nn.Conv2d(C, 1, 1)
        self.head3 = nn.Conv2d(C, 1, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        f5 = self.msce5(self.lat5(e5))
        f4 = self.msce4(self.lat4(e4))
        f3 = self.msce3(self.lat3(e3))
        f2 = self.lat2(e2)

        d4 = self.apfa4(f5, f4)
        d3 = self.apfa3(d4, f3)
        d2 = self.apfa2(d3, f2)

        out = F.interpolate(self.head(d2), (H, W), mode='bilinear', align_corners=False)
        if self.training:
            s4 = F.interpolate(self.head4(d4), (H, W), mode='bilinear', align_corners=False)
            s3 = F.interpolate(self.head3(d3), (H, W), mode='bilinear', align_corners=False)
            return out, s3, s4
        return out
