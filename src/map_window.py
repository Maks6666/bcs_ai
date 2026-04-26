import cv2
import numpy as np 

class MapWindow:
    def __init__(self, map_size: int, scale: int, flank_threshold: int):
        self.map_size = map_size
        self.center = map_size // 2
        self.scale = scale
        self.flank_threshold = flank_threshold

    

    def draw_screen(self):
        map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        scale = 1

        threshold = self.flank_threshold

        left_border = self.center - threshold * scale
        right_border = self.center + threshold * scale

        cv2.rectangle(map_img, (0, 0), (left_border, self.center), (50, 50, 100), -1)
        cv2.rectangle(map_img, (left_border, 0), (right_border, self.center), (50, 100, 50), -1)
        cv2.rectangle(map_img, (right_border, 0), (self.map_size, self.center), (100, 50, 50), -1)

        grid_step = self.flank_threshold

        for i in range(0, self.map_size, grid_step):
            cv2.line(map_img, (i, 0), (i, self.map_size), (40, 40, 40), 1)
            cv2.line(map_img, (0, i), (self.map_size, i), (40, 40, 40), 1)

        
        cv2.line(map_img, (0, self.center), (self.map_size, self.center), (80, 80, 80), 2)
        cv2.line(map_img, (self.center, 0), (self.center, self.map_size), (80, 80, 80), 2)

        cv2.circle(map_img, (self.center, self.center), 6, (255, 255, 255), -1)
        cv2.putText(map_img, "CAMERA", (self.center + 5, self.center - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return map_img

    
    def draw_objects(self, map_img, vehicles, positions, threat_scores, priority):
        for idx, (X, _, Z) in positions.items():
            # scale = 1

            px = int(self.center + X * self.scale)
            py = int(self.center - Z * self.scale)

            v_type = vehicles[idx]
            threat = threat_scores[idx]
            # tactic = tactics[idx] if idx in tactics else 'Analysing'

            colour = (0, 255, 0)
            if threat is not None:
                    
                if threat > 0.8:
                    colour = (0, 0, 255)
                elif threat > 0.5:
                    colour = (0, 165, 255)
                else:
                    colour = (0, 255, 0)

            text = f"{idx} | {v_type} | ({round(X, 2)}m {round(Z, 2)}m)"

            # --------------------------------------------------------------------------------------------------------------------

            # max_threat_idx = max(threat_scores, key=threat_scores.get)

            if idx == priority:
                cv2.circle(map_img, (px, py), 12, (0, 0, 255), 2)
                cv2.circle(map_img, (px, py), 5, (0, 0, 255), -1)
            
            # --------------------------------------------------------------------------------------------------------------------
                
            cv2.circle(map_img, (px, py), 5, colour, -1)
            cv2.putText(map_img, text, (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        return map_img