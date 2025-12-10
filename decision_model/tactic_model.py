import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

params_link = "/Users/maxkucher/PycharmProjects/bcs_ai_/decision_model/best_params.json"
with open(params_link, "r") as f:
    best_params = json.load(f)

# model = RandomForestClassifier(**best_params)
weight_link = "/Users/maxkucher/PycharmProjects/bcs_ai_/decision_model/model.pkl"

tactic_model = joblib.load(weight_link)
