import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

import os
_dir = os.path.dirname(__file__)

params_link = os.path.join(_dir, "best_params.json")
with open(params_link, "r") as f:
    best_params = json.load(f)

# model = RandomForestClassifier(**best_params)
weight_link = os.path.join(_dir, "model.pkl")

tactic_model = joblib.load(weight_link)
