import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bnorm = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.relu(self.bnorm(self.conv(x)))


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.bnorm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.bnorm2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.add_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=1),
                nn.BatchNorm2d(out_channels))
        else:
            self.add_conv = nn.Identity()

    def forward(self, x):
        out = self.bnorm1(self.conv1(x))
        add_out = self.add_conv(x)

        out = F.relu(out)
        out = self.bnorm2(self.conv2(out))

        out += add_out
        return out