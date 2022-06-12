import torch
import torch.nn as nn


from .unet_parts import OutConv
from .unet_model import UNet


class SyameseUNet(UNet):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__(n_channels, n_classes, bilinear)
        self.mixer = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y1 = self.inc(y)
        y2 = self.down1(y1)
        y3 = self.down2(y2)
        y4 = self.down3(y3)
        y5 = self.down4(y4)

        x5 = self.mixer(torch.abs(x5 - y5))

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
