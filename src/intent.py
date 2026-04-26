import pickle
from collections import defaultdict
import numpy as np 


class Inent:
    def __init__(self, history: defaultdict, intents: dict):
        self.history = history
        self.intents = intents
        self.ACTION_MAP = {
            "moving_back": 2,
            "center_flank": -2,
            "from_left_flank": -1,
            "from_right_flank": 1
        }


    def build_features(self, history):
        features = []

        # history - list of dicts
        for h in history:
            X, Y, Z = h['pos']
            vx, vy, speed = h["velocity"]
            distance = h["distance"]   
            threat = h["threat"]
            action = h["action"]


            motion_class = self.ACTION_MAP.get(action, 0)

            features.append([
                X, Z, 
                vx, vy, 
                speed, 
                motion_class, 
                distance, 
                threat
            ])

        return np.array(features).flatten()


    
    def calculate(self, idx):
        
        model_link = './intent_model/intent_model.pkl'
        with open(model_link, 'rb') as f:
            intent_model = pickle.load(f)


        if len(self.history[idx]) >= 10:
            history = list(self.history[idx])[-10:]
            features = self.build_features(history)

            intent = intent_model.predict([features])[0]
            self.intents[idx] = intent

            return intent
        
        else: 
            return None 