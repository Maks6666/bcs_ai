from math import e

class ThreatEstimator:
    def __init__(self, class_names):
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


    def score(self, class_id, distance, action, confidence):
        name = self.class_names[int(class_id)]

        type_score = self.type_weights.get(name, 0.5)

        if distance and distance > 0:
            distance_score = 1 / (1 + e ** -distance)
        else:
            distance_score = 0.0

        action_score = self.action_weights.get(action, 0.5)

        threat = (
            0.4 * type_score +
            0.3 * distance_score +
            0.2 * action_score +
            0.1 * confidence
        )

        return round(threat, 3)