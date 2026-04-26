from math import e
from math import sqrt
import pickle
import numpy as np

class ThreatEstimator:
    def __init__(self, class_names, max_dist):
        self.max_dist = max_dist
        self.class_names = class_names

        self.type_weights = {
            "TANK": 1.0,
            "IFV": 0.8,
            "APC": 0.7
        }

        self.action_weights = {
            "center_flank": 1.0,
            "from_left_flank": 0.9,
            "from_right_flank": 0.9,
            "frontal_attack": 0.8,
            "moving_back": 0.3
        }

        self.intent_weights = {
            "attack": 1.0,
            "reposition": 0.7,
            "retreat": 0.3,
            "idle": 0.1
        }


    def score(self, class_id, distance, action, action_proba, confidence, curr_pos, prev_pos, future_pos, intent):
        name = self.class_names[int(class_id)]
        # type score
        type_score = self.type_weights.get(name, 0.5)

        # ------------------------------------------------------------------------
        # distance score 
        if distance and distance > 0:
            distance_score = max(0, 1 - distance / self.max_dist)
        else:
            distance_score = 0.0
        
        # ------------------------------------------------------------------------
        # action score 
        action_score = self.action_weights.get(action, 0.5)
        if action == "Analyzing...":
            action_weight = 0
        else:
            action_weight = 0.15

        if action_proba is None:
            action_proba = 0.0
        
        action_score *= float(action_proba)


        # ------------------------------------------------------------------------
        # direction_score
        if prev_pos is not None:
            dx = curr_pos[0] - prev_pos[0]
            dz = curr_pos[1] - prev_pos[1]
        else:
            dx, dz = 0, 0

        movement = sqrt(dx ** 2 + dz ** 2)

        if movement < 0.2:
            direction_score = 0.5
        elif dz < 0:
            direction_score = 1.0
        elif abs(dx) > abs(dz):
            direction_score = 0.7
        else:
            direction_score = 0.3

        # ------------------------------------------------------------------------
        # future distance score
        if future_pos:
            _, Z_future = future_pos
            future_dist = Z_future
            future_dist_score = max(0, 1 - future_dist / self.max_dist)

        else:
            future_dist_score = 0

        # ------------------------------------------------------------------------
        # intent score

        intent_score = self.intent_weights.get(intent, 0.5)

        # threat = (
        #     0.2 * type_score +
        #     0.15 * distance_score +
        #     0.15 * future_dist_score + 
        #     action_weight * action_score +
        #     0.1 * direction_score +
        #     0.1 * confidence +
        #     0.2 * intent_score
        # )

        model_link = './threat_model/threat_model.pkl'
        with open(model_link, 'rb') as file:
            model = pickle.load(file)


        data = [[type_score, distance_score, future_dist_score, action_score, direction_score, confidence, intent_score]]
        data = np.array(data)
        
        threat = model.predict(data)


        return round(float(threat), 3)