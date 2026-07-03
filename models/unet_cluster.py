import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetCluster(nn.Module):
    """
    U-Net dense clustering model.

    Output:
    - logits: raw cluster logits
    - probs: soft cluster assignments
    - recon: reconstruction using learned cluster prototypes
    """

    def __init__(self, in_channels=1, num_clusters=4, base_channels=32):
        super().__init__()

        b = base_channels

        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)

        self.pool = nn.MaxPool2d(2)

        self.center = ConvBlock(b * 8, b * 16)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, 2)
        self.dec4 = ConvBlock(b * 16, b * 8)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2)
        self.dec3 = ConvBlock(b * 8, b * 4)

        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2)
        self.dec2 = ConvBlock(b * 4, b * 2)

        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, 2)
        self.dec1 = ConvBlock(b * 2, b)

        self.cluster_head = nn.Conv2d(b, num_clusters, kernel_size=1)

        # learned intensity prototypes for reconstruction
        init = torch.linspace(-1.0, 1.0, steps=num_clusters).view(1, num_clusters, 1, 1)
        self.prototypes = nn.Parameter(init)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        c = self.center(self.pool(e4))

        d4 = self.up4(c)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.cluster_head(d1)
        probs = F.softmax(logits, dim=1)
        recon = torch.sum(probs * self.prototypes, dim=1, keepdim=True)

        return logits, probs, recon
