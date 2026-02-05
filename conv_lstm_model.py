import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from models.constructor import ConvBlock, ResConvBlock
import torchvision.models.video as video_models


# class TeacherNet(nn.Module):
#     def __init__(self, num_layers=2, hidden_size=512, feature_dim=2048, bidirectional=True, num_classes=4):
#         super().__init__()

#         base_model = resnet50(weights="IMAGENET1K_V2")
#         self.cnn = nn.Sequential(*list(base_model.children())[:-2])
#         self.aap = nn.AdaptiveAvgPool2d((1, 1))

#         self.feature_dim = feature_dim

#         self.lstm = nn.LSTM(
#             input_size=self.feature_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=bidirectional
#         )

#         lstm_output = hidden_size * (2 if bidirectional else 1)

#         self.classifier = nn.Sequential(
#             nn.Linear(lstm_output, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)

#         )

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         x = x.view(B * T, C, H, W)

#         feats = self.cnn(x)
#         feats = self.aap(feats).flatten(1)

#         feats = feats.view(B, T, -1)

#         lstm_out, _ = self.lstm(feats)
#         lstm_out = lstm_out.mean(dim=1)

#         out = self.classifier(lstm_out)
#         return out

#     def predict(self, x):
#         self.eval()

#         if len(x.shape) == 4:
#             x = x.unsqueeze(0)

#         with torch.no_grad():
#             out = self.forward(x)

#         out = torch.softmax(out, dim=1)
#         res = torch.argmax(out, dim=1)

#         return res.item()

class TeacherNet(nn.Module):
    def __init__(self, classes=4):
        super().__init__()

        self.model = video_models.r3d_18(weights="KINETICS400_V1")
        self.model.fc = nn.Linear(512, classes)

    def forward(self, x):
        out = self.model(x)
        return out

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            out = self.forward(x)
        out = torch.argmax(out, axis=1)
        return out.item()



# class StudentNet(nn.Module):
#     def __init__(self, num_layers=1, channels=[32, 64, 128, 256, 512, 512], input_dim=512, hidden_dim=256,
#                  bidirectional=True, num_classes=4):
#         super().__init__()

#         self.layer_list = nn.ModuleList()
#         self.channels = channels

#         in_channel = 3
#         for channel in self.channels:
#             layer = ResConvBlock(in_channel, channel, 3, 2 if in_channel != channel else 1, 1)
#             in_channel = channel
#             self.layer_list.append(layer)

#         self.aap = nn.AdaptiveAvgPool2d((1, 1))

#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=bidirectional,
#             dropout=0.3
#         )

#         lstm_out = hidden_dim * (2 if bidirectional == True else 1)

#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(lstm_out, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         B, F, C, H, W = x.shape
#         x = x.view(B * F, C, H, W)

#         for layer in self.layer_list:
#             x = layer(x)

#         out = self.aap(x).flatten(1)
#         out = out.view(B, F, -1)

#         out, (_, _) = self.lstm(out)
#         out = out.mean(dim=1)

#         res = self.classifier(out)

#         return res

#     def predict(self, x):
#         self.eval()

#         if len(x.shape) == 4:
#             x = x.unsqueeze(0)

#         with torch.no_grad():
#             out = self.forward(x)

#         out = torch.softmax(out, dim=1)
#         out = torch.argmax(out, dim=1)

#         return out.item()


class StudentNet(nn.Module):
    def __init__(self, num_layers=1, channels=[32, 64, 128, 256, 512, 1024], input_dim=1024, hidden_dim=256,
                 bidirectional=True, num_classes=4):
        super().__init__()

        self.layer_list = nn.ModuleList()
        self.channels = channels

        in_channel = 3
        for channel in self.channels:
            layer = ConvBlock(in_channel, channel, 4, 2, 1)
            in_channel = channel
            self.layer_list.append(layer)

        self.aap = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_out = hidden_dim * (2 if bidirectional == True else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.ReLU(),
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


# class StudentNet(nn.Module):
#     def __init__(self, num_layers=1, input_dim=512, hidden_dim=256, bidirectional=True, num_classes=4): 
#         super().__init__()

#         body = models.resnet34(weights='DEFAULT')
#         self.cnn = nn.Sequential(*list(body.children())[:-2])
#         self.aap = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()

#         self.lstm = nn.LSTM(
#             input_size = input_dim,
#             hidden_size = hidden_dim, 
#             num_layers = num_layers,
#             batch_first = True, 
#             bidirectional = bidirectional,
#             dropout=0.3
#         )

#         lstm_out = hidden_dim * (2 if bidirectional == True else 1)

#         self.classifier = nn.Sequential(
#             nn.Linear(lstm_out, 128),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(128, num_classes)
#         )


#     def forward(self, x):
#         B, F, C, H, W = x.shape
#         x = x.view(B*F, C, H, W)
        
#         out = self.cnn(x)
#         out = self.aap(out)
#         out = self.flatten(out)
        
#         out = out.view(B, F, -1)

#         out, (_, _) = self.lstm(out)
#         out = out.mean(dim=1)

#         res = self.classifier(out)
        
#         return res

#     def predict(self, x):
#         self.eval()

#         if len(x.shape) == 4:
#             x = x.unsqueeze(0)

#         with torch.no_grad():
#             out = self.forward(x)

#         out = torch.softmax(out, dim=1)
#         out = torch.argmax(out, dim=1)

#         return out.item()


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        weight = self.attn(x)
        weight = torch.softmax(weight, dim=1)

        return (x * weight).sum(dim=1) 


class Net(nn.Module):
    def __init__(self, layers=3, hidden_size=256, feature_dim = 3, bidirectional = True, num_classes = 4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = feature_dim,
            hidden_size = hidden_size,
            num_layers = layers,
            batch_first = True,
            bidirectional = bidirectional
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        self.ta = TemporalAttention(out_dim)
        self.classifier = nn.Linear(out_dim, num_classes) 

    def forward(self, x):
        lstm_out, (_, _) = self.lstm(x)
        out = self.ta(lstm_out)

        logits = self.classifier(out)
        return logits

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            out = self.forward(x)

        out = torch.argmax(out, dim=1)
        return out.item()




device =  "mps" if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net()


link = './models/optical_tactic_model.pt'
model.load_state_dict(torch.load(link, map_location=device))
model.to(device)

print(f"Tactics tracking model successfully loaded on device: {device}")
