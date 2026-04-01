from math import e
from math import sqrt

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


    def score(self, class_id, distance, action, confidence, curr_pos, prev_pos, future_pos):
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


        # ------------------------------------------------------------------------
        # direction_score
        if prev_pos is not None:
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
        else:
            dx, dy = 0, 0

        movement = sqrt(dx ** 2 + dy ** 2)

        if movement < 0.2:
            direction_score = 0.5
        elif dy < 0:
            direction_score = 1.0
        elif abs(dx) > abs(dy):
            direction_score = 0.7
        else:
            direction_score = 0.3

        # ------------------------------------------------------------------------
        # future distance score
        if future_pos:
            _, Y_future = future_pos
            future_dist = Y_future
            future_dist_score = max(0, 1 - future_dist / self.max_dist)

        else:
            future_dist_score = 0

        threat = (
            0.25 * type_score +
            0.2 * distance_score +
            0.2 * future_dist_score + 
            action_weight * action_score +
            0.1 * direction_score +
            0.1 * confidence
        )

        return round(threat, 3)