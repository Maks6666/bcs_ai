import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from models.constructor import ConvBlock, ResConvBlock
import torchvision.models.video as video_models


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
    
    def predict_proba(self, x):
        self.eval()

        with torch.no_grad():
            out = self.forward(x)
        
        out = torch.max(torch.softmax(out, dim=1))
    

        return out 




device =  "mps" if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net()


link = './models/optical_tactic_model.pt'
model.load_state_dict(torch.load(link, map_location=device))
model.to(device)

print(f"Tactics tracking model successfully loaded on device: {device}")


# tensor = torch.randn(1, 15, 3).to(device)

# out = model.predict_proba(tensor)

# print(out)

