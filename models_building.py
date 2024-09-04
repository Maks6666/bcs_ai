# deploy custom models

import torch
import numpy as np
from fontTools.misc.cython import returns
from torch import nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"


class VehiclesModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.linear1 = nn.Linear(in_features=7, out_features=28)
        self.bnorm1 = nn.BatchNorm1d(28)
        self.linear2 = nn.Linear(in_features=28, out_features=24)
        self.drop1 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(in_features=24, out_features=16)
        self.linear4 = nn.Linear(in_features=16, out_features=8)
        self.linear5 = nn.Linear(in_features=8, out_features=num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.bnorm1(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        return out

    def predict(self, x):
        x = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            y_pred = F.softmax(self.forward(x), dim=-1)

        return y_pred


model = VehiclesModel().to(device)
model.load_state_dict(torch.load("custom_models/vehicles.pt", map_location=device))
model.to(device)
model.eval()



def vehicles_model(data):
    model = VehiclesModel().to(device)
    model.load_state_dict(torch.load("custom_models/vehicles.pt", map_location=device))
    model.to(device)
    model.eval()

    x = torch.FloatTensor(data).to(device)
    x = x.unsqueeze(0)

    with torch.no_grad():
        res = model(x)

    t_x = torch.argmax(res)
    t_array = np.array(t_x)

    return t_array

# ----------------------------------------------------------------------------------------------------------------------

class SpecialModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.linear1 = nn.Linear(in_features=6, out_features=24)
        self.bnorm1 = nn.BatchNorm1d(24)
        self.linear2 = nn.Linear(in_features=24, out_features=12)
        self.linear3 = nn.Linear(in_features=12, out_features=12)
        self.linear4 = nn.Linear(in_features=12, out_features=6)
        self.linear5 = nn.Linear(in_features=6, out_features=num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.bnorm1(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)

        return out

    def predict(self, x, device="cpu"):
        x = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            y_pred = torch.softmax(self.forward(x), dim=-1)

        return y_pred


model = SpecialModel().to(device)
model.load_state_dict(torch.load("custom_models/specials.pt", map_location=device))
model.to(device)
model.eval()

def special_model(data):
    model = SpecialModel().to(device)
    model.load_state_dict(torch.load("custom_models/specials.pt", map_location=device))
    model.to(device)
    model.eval()

    x = torch.FloatTensor(data).to(device)
    x = x.unsqueeze(0)

    with torch.no_grad():
        res = model(x)

    t_x = torch.argmax(res)
    t_array = np.array(t_x)

    return t_array

# ----------------------------------------------------------------------------------------------------------------------

class AviationModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.linear1 = nn.Linear(in_features=6, out_features=24)
        self.bnorm1 = nn.BatchNorm1d(num_features=24)
        self.linear2 = nn.Linear(in_features=24, out_features=24)
        self.drop1 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(in_features=24, out_features=12)
        self.linear4 = nn.Linear(in_features=12, out_features=num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.bnorm1(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)

        return out

    def predict(self, x, device="cpu"):
        x = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            y_pred = F.softmax(self.forward(x), dim=-1)

        return y_pred


model = AviationModel().to(device)
model.load_state_dict(torch.load("custom_models/aviation.pt", map_location=device))
model.to(device)
model.eval()

def aviation_model(data):
    model = AviationModel().to(device)
    model.load_state_dict(torch.load("custom_models/aviation.pt", map_location=device))
    model.to(device)
    model.eval()

    x = torch.FloatTensor(data).to(device)
    x = x.unsqueeze(0)

    with torch.no_grad():
        res = model(x)

    t_x = torch.argmax(res)
    t_array = np.array(t_x)

    return t_array



# ----------------------------------------------------------------------------------------------------------------------
class ArtilleryModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.linear1 = nn.Linear(in_features=5, out_features=28)
        self.bnorm1 = nn.BatchNorm1d(28)
        self.linear2 = nn.Linear(in_features=28, out_features=14)
        self.drop1 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(in_features=14, out_features=7)
        self.linear4 = nn.Linear(in_features=7, out_features=5)
        self.linear5 = nn.Linear(in_features=5, out_features=num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.bnorm1(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)

        return out

    def predict(self, x):
        x = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            y_pred = F.softmax(self.forward(x), dim=-1)

        return y_pred


model = ArtilleryModel().to(device)
model.load_state_dict(torch.load("custom_models/artillery.pt", map_location=device))
model.to(device)
model.eval()

def artillery_model(data):
    model = ArtilleryModel().to(device)
    model.load_state_dict(torch.load("custom_models/artillery.pt", map_location=device))
    model.to(device)
    model.eval()

    x = torch.FloatTensor(data).to(device)
    x = x.unsqueeze(0)

    with torch.no_grad():
        res = model(x)

    t_x = torch.argmax(res)
    t_array = np.array(t_x)

    return t_array


class TroopsModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.linear1 = nn.Linear(in_features=4, out_features=24)
        self.bnorm1 = nn.BatchNorm1d(24)
        self.linear2 = nn.Linear(in_features=24, out_features=24)
        self.drop1 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(in_features=24, out_features=24)
        self.linear4 = nn.Linear(in_features=24, out_features=12)
        self.linear5 = nn.Linear(in_features=12, out_features=6)
        self.linear6 = nn.Linear(in_features=6, out_features=num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.bnorm1(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        out = F.relu(out)
        out = self.linear6(out)
        return out

    def predict(self, x):
        x = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            y_pred = F.softmax(self.forward(x), dim=-1)

        return y_pred


model = TroopsModel().to(device)
model.load_state_dict(torch.load("custom_models/troops.pt", map_location=device))
model.to(device)
model.eval()

def troops_model(data):
    model = TroopsModel().to(device)
    model.load_state_dict(torch.load("custom_models/troops.pt", map_location=device))
    model.to(device)
    model.eval()

    x = torch.FloatTensor(data).to(device)
    x = x.unsqueeze(0)

    with torch.no_grad():
        res = model(x)

    t_x = torch.argmax(res)
    t_array = np.array(t_x)

    return t_array
