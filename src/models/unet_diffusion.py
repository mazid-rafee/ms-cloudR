import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )

        self.mid = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU()
        )

        self.out = nn.Conv2d(64, 13, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        m = self.mid(e2)
        d = self.dec1(m)
        return self.out(d)
