import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from constructor import ConvBlock, ResConvBlock


class TeacherNet(nn.Module):
    def __init__(self, num_layers=2, hidden_size=512, feature_dim=2048, bidirectional=True, num_classes=4):
        super().__init__()

        base_model = resnet50(weights="IMAGENET1K_V2")
        self.cnn = nn.Sequential(*list(base_model.children())[:-2])
        self.aap = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim = feature_dim

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_output = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)

        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.cnn(x)
        feats = self.aap(feats).flatten(1)

        feats = feats.view(B, T, -1)

        lstm_out, _ = self.lstm(feats)
        lstm_out = lstm_out.mean(dim=1)

        out = self.classifier(lstm_out)
        return out

    def predict(self, x):
        self.eval()

        if len(x.shape) == 4:
            x = x.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(x)

        out = torch.softmax(out, dim=1)
        res = torch.argmax(out, dim=1)

        return res.item()


# class StudentNet(nn.Module):
#     def __init__(self, num_layers=1, channels=[32, 64, 128, 256, 512, 1024], input_dim=1024, hidden_dim=256,
#                  bidirectional=True, num_classes=4):
#         super().__init__()
#
#         self.layer_list = nn.ModuleList()
#         self.channels = channels
#
#         in_channel = 3
#         for channel in self.channels:
#             layer = ConvBlock(in_channel, channel, 4, 2, 1)
#             in_channel = channel
#             self.layer_list.append(layer)
#
#         self.aap = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=bidirectional
#         )
#
#         lstm_out = hidden_dim * (2 if bidirectional == True else 1)
#
#         self.classifier = nn.Sequential(
#             nn.Linear(lstm_out, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, x):
#         B, F, C, H, W = x.shape
#         x = x.view(B * F, C, H, W)
#
#         for layer in self.layer_list:
#             x = layer(x)
#
#         out = self.aap(x).flatten(1)
#         out = out.view(B, F, -1)
#
#         out, (_, _) = self.lstm(out)
#         out = out.mean(dim=1)
#
#         res = self.classifier(out)
#
#         return res
#
#     def predict(self, x):
#         self.eval()
#
#         if len(x.shape) == 4:
#             x = x.unsqueeze(0)
#
#         with torch.no_grad():
#             out = self.forward(x)
#
#         out = torch.softmax(out, dim=1)
#         out = torch.argmax(out, dim=1)
#
#         return out.item()

class StudentNet(nn.Module):
    def __init__(self, num_layers=1, channels=[32, 64, 128, 256, 512, 512], input_dim=512, hidden_dim=256,
                 bidirectional=True, num_classes=4):
        super().__init__()

        self.layer_list = nn.ModuleList()
        self.channels = channels

        in_channel = 3
        for channel in self.channels:
            layer = ResConvBlock(in_channel, channel, 3, 2 if in_channel != channel else 1, 1)
            in_channel = channel
            self.layer_list.append(layer)

        self.aap = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3
        )

        lstm_out = hidden_dim * (2 if bidirectional == True else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_out, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)

        for layer in self.layer_list:
            x = layer(x)

        out = self.aap(x).flatten(1)
        out = out.view(B, F, -1)

        out, (_, _) = self.lstm(out)
        out = out.mean(dim=1)

        res = self.classifier(out)

        return res

    def predict(self, x):
        self.eval()

        if len(x.shape) == 4:
            x = x.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(x)

        out = torch.softmax(out, dim=1)
        out = torch.argmax(out, dim=1)

        return out.item()


class StudentGRUNet(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256, 512, 512], C=512, num_classes=4):
        super().__init__()

        self.layer_list = nn.ModuleList()
        self.channels = channels

        input_channel = 3
        for channel in self.channels:
            layer = ResConvBlock(input_channel, channel, 3, 2 if input_channel != channel else 1, 1)
            self.layer_list.append(layer)
            input_channel = channel

        self.aap = nn.AdaptiveAvgPool2d((1, 1))

        self.gru = nn.GRU(
            input_size=C,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)

        )

    def forward(self, x):
        B, R, C, H, W = x.shape
        x = x.view(B * R, C, H, W)
        for layer in self.layer_list:
            x = layer(x)

        out = self.aap(x)
        out = out.flatten(1)
        out = out.view(B, R, -1)

        out_h, _ = self.gru(out)
        out = out_h.mean(1)

        out = self.classifier(out)

        return out

    def predict(self, x):
        self.eval()

        if len(x.shape) == 4:
            x = x.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(x)

        out = torch.softmax(out, dim=1)
        res = torch.argmax(out, dim=1).item()

        return res


device = "cuda" if torch.cuda.is_available() else "cpu"
# model = TeacherNet()
model = StudentNet()
# model = StudentGRUNet()
model.to()

print(f"Tactics tracking model successfully loaded on device: {device}")

# link = "./models/gru_model_v02.pt"
link = "./models/tactic_model_v02.pt"
model.load_state_dict(torch.load(link, map_location=device))
# print(next(model.parameters()).dtype)

