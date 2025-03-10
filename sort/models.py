import torch
from torch import nn
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"


class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool = True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size=3, padding=1)

        if pool == True:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = None

        self.add_conv = nn.Sequential()

        if in_channels != out_channels:
            self.add_conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, padding = 1)

    def forward(self, x):
        out = self.conv1(x)
        add_out = self.add_conv(x)

        # out = F.relu(out)

        out = self.conv2(out)

        out += add_out

        out = F.relu(out)

        if self.pool:
            out = self.pool(out)

        return out


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(out)

        fin_out = torch.concat([x, out], dim=1)

        return fin_out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, pool=True):
        super().__init__()

        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        if pool == True:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = None

        for i in range(self.num_layers):
            self.layers.append(DenseLayer(in_channels, growth_rate))
            in_channels += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

            if self.pool:
                x = self.pool(x)

        return x


# 1st version:

# class Oko(nn.Module):
#     def __init__(self, num_classes=3):
#         super().__init__()
#
#         self.skip_block1 = SkipBlock(3, 32)
#
#         self.dense_block2 = DenseBlock(32, 16, num_layers=4)
#
#         self.dense_block3 = DenseBlock(96, 16, num_layers=3, pool=False)
#
#         self.dense_block4 = DenseBlock(144, 16, num_layers=1, pool=True)
#
#         # 160
#
#         self.skip_block5 = SkipBlock(160, 256)
#
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#
#         self.flatten = nn.Flatten()
#
#         self.linear1 = nn.Linear(512, 64)
#         self.linear2 = nn.Linear(64, num_classes)
#
#     def forward(self, x):
#         out = self.skip_block1(x)
#
#         out = self.dense_block2(out)
#         out = self.dense_block3(out)
#         out = self.dense_block4(out)
#         out = self.skip_block5(out)
#         out = self.conv5(out)
#
#         out = self.flatten(out)
#         out = self.linear1(out)
#         out = F.relu(out)
#         res = self.linear2(out)
#
#         return res
#
#     def predict(self, x):
#         out = None
#         self.eval()
#
#         with torch.no_grad():
#
#             if len(x.shape) == 4:
#                 x = x.to(device)
#                 out = self.forward(x)
#
#             if len(x.shape) == 3:
#                 x = x.unsqueeze(0).to(device)
#                 out = self.forward(x)
#
#             t_res = torch.softmax(out, dim=1)
#             res = torch.argmax(t_res, dim=1)
#             return res


# 2nd version:

# class Oko(nn.Module):
#     def __init__(self, num_classes=3):
#         super().__init__()
#
#         self.skip_block1 = SkipBlock(3, 32)
#
#         self.dense_block2 = DenseBlock(32, 16, num_layers=4)
#
#         self.dense_block3 = DenseBlock(96, 16, num_layers=3, pool=False)
#
#         self.dense_block4 = DenseBlock(144, 16, num_layers=1, pool=True)
#
#         # 160
#
#         self.skip_block5 = SkipBlock(160, 256)
#
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
#
#         self.flatten = nn.Flatten()
#
#         self.linear1 = nn.Linear(1024, 256)
#         self.drop = nn.Dropout(0.2)
#         self.linear2 = nn.Linear(256, 64)
#         self.linear3 = nn.Linear(64, num_classes)
#
#     def forward(self, x):
#         out = self.skip_block1(x)
#
#         out = self.dense_block2(out)
#         out = self.dense_block3(out)
#         out = self.dense_block4(out)
#         out = self.skip_block5(out)
#         out = self.conv5(out)
#         out = F.relu(out)
#         out = self.conv6(out)
#
#         out = self.flatten(out)
#         out = self.linear1(out)
#         out = F.relu(out)
#         out = self.drop(out)
#         out = self.linear2(out)
#         out = F.relu(out)
#         res = self.linear3(out)
#
#         return res
#
#     def predict(self, x):
#         out = None
#         self.eval()
#
#         with torch.no_grad():
#
#             if len(x.shape) == 4:
#                 x = x.to(device)
#                 out = self.forward(x)
#
#             if len(x.shape) == 3:
#                 x = x.unsqueeze(0).to(device)
#                 out = self.forward(x)
#
#             t_res = torch.softmax(out, dim=1)
#             res = torch.argmax(t_res, dim=1)
#             return res


# 3rd version:

class SEBlock(nn.Module):
    def __init__(self, C, r):
        super().__init__()

        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(C, C//r)
        self.linear2 = nn.Linear(C//r, C)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.aap(x)
        out = self.flatten(out)

        out = self.linear1(out)
        out = self.relu(out)

        out = self.linear2(out)
        out = self.sigmoid(out)

        out = out[:, :, None, None]
        res = out * x

        return res



class Oko(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.skip_block1 = SkipBlock(3, 32)

        self.dense_block2 = DenseBlock(32, 16, num_layers=4)

        self.se_block2_1 = SEBlock(C=96, r=16)

        self.dense_block3 = DenseBlock(96, 16, num_layers=3, pool=False)

        self.se_block3_1 = SEBlock(C=144, r=16)

        self.dense_block4 = DenseBlock(144, 16, num_layers=1, pool=True)

        self.se_block4_1 = SEBlock(C=160, r=16)

        # 160

        self.skip_block5 = SkipBlock(160, 256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(1024, 256)
        self.drop = nn.Dropout(0.2)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.skip_block1(x)

        out = self.dense_block2(out)
        out = self.se_block2_1(out)

        out = self.dense_block3(out)
        out = self.se_block3_1(out)

        out = self.dense_block4(out)
        out = self.se_block4_1(out)

        out = self.skip_block5(out)
        out = self.conv5(out)
        out = F.relu(out)
        out = self.conv6(out)

        out = self.flatten(out)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.drop(out)
        out = self.linear2(out)
        out = F.relu(out)
        res = self.linear3(out)

        return res

    def predict(self, x):
        print("Predicting...")
        out = None
        self.eval()

        with torch.no_grad():

            if len(x.shape) == 4:
                x = x.to(device)
                out = self.forward(x)

            if len(x.shape) == 3:
                x = x.unsqueeze(0).to(device)
                out = self.forward(x)

            t_res = torch.softmax(out, dim=1)
            res = torch.argmax(t_res, dim=1)
            return res


model = Oko()
model.to(device)

model.load_state_dict(torch.load("classification_detection/classification_weights/oko_updated_2.pt", map_location=device))


tensor = torch.rand((1, 3, 224, 224)).to(device)
res = model.predict(tensor)

if res is not None:
    print("All dependencies match!")




