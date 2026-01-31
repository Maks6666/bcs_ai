
from decision_model.tactic_model import tactic_model
import numpy as np

class TacticPredictor:
    def __init__(self, maneuvers: list):
        self.maneuvers = maneuvers
    def predict_tactic(self, actions, tactics: dict):
        moving_forward_num = 0
        from_left_flank_num = 0
        from_right_flank_num = 0
        move_back_num = 0

        if actions and len(tactics) > 0:
            moving_forward, from_left_flank, from_right_flank, moving_back = actions

            if 0 < moving_forward < 5:
                moving_forward_num = 1
            elif moving_forward > 5:
                moving_forward_num = 2


            if 0 < from_left_flank < 5:
                from_left_flank_num = 1
            elif from_left_flank > 5:
                from_left_flank_num = 2


            if 0 < from_right_flank < 5:
                from_right_flank_num = 1
            elif from_right_flank > 5:
                from_right_flank_num = 2

            if 0 < moving_back < 5:
                move_back_num = 1
            elif moving_back > 5:
                move_back_num = 2

            array = np.array([[moving_forward_num, from_right_flank_num, from_left_flank_num, move_back_num]])
            pred = tactic_model.predict(array)

            res = self.maneuvers[int(pred)]

            return res

        else:
            return "Analyzing..."
