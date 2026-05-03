import torch 
from torch import nn 


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        weights = self.attn(x)
        weights = torch.softmax(weights, dim=1)

        return (weights * x).sum(dim=1)
    

class Net(nn.Module):
    def __init__(self, layers=3, hidden_size=256, feature_dim=12, bidirectional=True, num_classes=5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_dim,
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
        out = self.classifier(out)

        return out

    def predict(self, x):
        self.eval()

        with torch.no_grad():
            out = self.forward(x)

        res = torch.argmax(out, dim=1)
        return res.item()
    
    def predict_proba(self, x):
        self.eval()

        with torch.no_grad():
            out = self.forward(x)

        proba = torch.softmax(out, dim=1)
        proba = proba.max(dim=1).values
        return float(proba.cpu().numpy())
        
    
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net()


model_link = './infantry_tactics_model/infantry_model.pt'
model.load_state_dict(torch.load(model_link, map_location=device))