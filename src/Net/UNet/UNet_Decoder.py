
import torch.nn as nn

class UNetDecode(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, kaiming_initialization=False):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.decode(x)