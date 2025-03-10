import torch
from torch import nn
import torch.nn.functional as F

class SE_block(nn.Module):
    def __init__(self, C, r=16):
        super().__init__()

        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(C, C//r)
        self.linear2 = nn.Linear(C//r, C)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.aap(x)
        out = self.flatten(out)

        out = self.linear1(out)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.sigmoid(out)

        out = out[:, :, None, None]

        res = x * out

        return res


class SE_Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bnorm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bnorm2 = nn.BatchNorm2d(out_channels)

        self.add_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels)

        )
        # self.add_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.se_block = SE_block(out_channels)

    def forward(self, x):
        out = F.leaky_relu(self.bnorm1(self.conv1(x)))
        add_out = self.add_conv(x)

        # out = F.leaky_relu(out)
        out = F.leaky_relu(self.bnorm2(self.conv2(out)))
        # out = self.bnorm2(out)
        out = self.se_block(out)

        out += add_out

        return out


class Oko(nn.Module):
    def __init__(self, outputs=3):
        super().__init__()

        self.conv1 = SE_Res_block(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = SE_Res_block(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = SE_Res_block(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = SE_Res_block(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = SE_Res_block(128, 128, kernel_size=4, stride=2, padding=1)

        self.conv6 = SE_Res_block(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv7 = SE_Res_block(256, 256, kernel_size=4, stride=2, padding=1)

        self.conv8 = SE_Res_block(256, 512, kernel_size=4, stride=2, padding=1)

        # self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # self.aap = nn.AdaptiveAvgPool((1, 1))

        self.linear1 = nn.Linear(512, 128)
        self.lin_bnorm1 = nn.BatchNorm1d(128)

        self.linear2 = nn.Linear(128, 64)
        self.lin_bnorm2 = nn.BatchNorm1d(64)

        self.linear3 = nn.Linear(64, 16)
        self.lin_bnorm3 = nn.BatchNorm1d(16)

        self.linear4 = nn.Linear(16, outputs)
        # self.lin_bnorm4 = nn.BatchNorm1d(outputs)

    def forward(self, x):
        out1 = self.conv1(x)
        # print(out1.shape)
        out2 = self.conv2(out1)
        # print(out2.shape)
        out3 = self.conv3(out2)
        # print(out3.shape)

        out4 = self.conv4(out3)
        # print(out4.shape)
        out5 = self.conv5(out4)
        # print(out5.shape)

        out6 = self.conv6(out5)
        # print(out6.shape)
        out7 = self.conv7(out6)
        # print(out7.shape)
        out8 = self.conv8(out7)
        # print(out8.shape)

        # out7 = self.aap(out7)
        out = self.flatten(out8)

        out = F.leaky_relu(self.lin_bnorm1(self.linear1(out)))
        out = F.leaky_relu(self.lin_bnorm2(self.linear2(out)))
        out = F.leaky_relu(self.lin_bnorm3(self.linear3(out)))
        res = self.linear4(out)
        return res

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            if len(x.shape) == 3:
                x = x.unsqueeze(0)

            t_res = self.forward(x)
            t_res = torch.softmax(t_res, dim=-1)
            res = torch.argmax(t_res).item()

        return res


def load_model(device):
    model = Oko()
    model.to(device)
    model.load_state_dict(torch.load("models/oko_classifier.pt", map_location=device))
    print("Model successfully loaded!")
    return model


