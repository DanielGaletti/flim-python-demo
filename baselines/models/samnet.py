"""
SAMNet: Stereoscopically Attentive Multi-scale Network
Reimplementação baseada em: Liu et al., IEEE TIP 2021
Backbone: MobileNetV2 (ImageNet pretrained)
~1.33M params no decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBnRelu(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=1, dilation=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, p, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class SAMModule(nn.Module):
    """Stereoscopically Attentive Multi-scale module"""
    def __init__(self, channels):
        super().__init__()
        self.branch1 = ConvBnRelu(channels, channels // 4, k=1, p=0)
        self.branch2 = ConvBnRelu(channels, channels // 4, k=3, p=1)
        self.branch3 = ConvBnRelu(channels, channels // 4, k=3, p=3, dilation=3)
        self.branch4 = ConvBnRelu(channels, channels // 4, k=3, p=5, dilation=5)
        self.fuse = ConvBnRelu(channels, channels, k=1, p=0)
        # Spatial attention
        self.attn = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = self.fuse(torch.cat([b1, b2, b3, b4], dim=1))
        return out * self.attn(out) + x


class DecoderBlock(nn.Module):
    def __init__(self, cin, skip_c, cout):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBnRelu(cin + skip_c, cout)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SAMNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        mv2 = models.mobilenet_v2(pretrained=pretrained)
        features = mv2.features

        # Encoder stages (MobileNetV2 feature maps)
        self.enc1 = features[0:2]    # stride 2, 16ch
        self.enc2 = features[2:4]    # stride 4, 24ch
        self.enc3 = features[4:7]    # stride 8, 32ch
        self.enc4 = features[7:14]   # stride 16, 96ch
        self.enc5 = features[14:19]  # stride 32, 1280ch

        # Reduce channels
        self.lat5 = ConvBnRelu(1280, 128, k=1, p=0)
        self.lat4 = ConvBnRelu(96,   64,  k=1, p=0)
        self.lat3 = ConvBnRelu(32,   32,  k=1, p=0)
        self.lat2 = ConvBnRelu(24,   16,  k=1, p=0)

        # SAM modules
        self.sam5 = SAMModule(128)
        self.sam4 = SAMModule(64)

        # Decoder
        self.dec4 = DecoderBlock(128, 64, 64)
        self.dec3 = DecoderBlock(64,  32, 32)
        self.dec2 = DecoderBlock(32,  16, 16)
        self.dec1 = DecoderBlock(16,   0, 16)

        # Saliency heads (deep supervision)
        self.head5 = nn.Conv2d(128, 1, 1)
        self.head4 = nn.Conv2d(64,  1, 1)
        self.head3 = nn.Conv2d(32,  1, 1)
        self.head  = nn.Conv2d(16,  1, 1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        f5 = self.sam5(self.lat5(e5))
        f4 = self.sam4(self.lat4(e4))
        f3 = self.lat3(e3)
        f2 = self.lat2(e2)

        d4 = self.dec4(f5, f4)
        d3 = self.dec3(d4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2)

        out = F.interpolate(self.head(d1), (H, W), mode='bilinear', align_corners=False)
        if self.training:
            s5 = F.interpolate(self.head5(f5), (H, W), mode='bilinear', align_corners=False)
            s4 = F.interpolate(self.head4(d4), (H, W), mode='bilinear', align_corners=False)
            s3 = F.interpolate(self.head3(d3), (H, W), mode='bilinear', align_corners=False)
            return out, s3, s4, s5
        return out
