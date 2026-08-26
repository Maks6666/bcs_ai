import joblib
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

#
# params_link = "/Users/maxkucher/PycharmProjects/bcs_ai_/weapon_model/best_params.json"
# with open(params_link) as f:
#     best_params = json.load(f)

import os
_dir = os.path.dirname(__file__)

weight_link = os.path.join(_dir, "model.pkl")

weapon_model = joblib.load(weight_link)

