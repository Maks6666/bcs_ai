import cv2
import numpy as np
import math

class MapWindow:
    def __init__(self, map_size: int, scale: int, flank_threshold: int, cameras=None):
        self.map_size = map_size
        self.center = map_size // 2
        self.scale = scale
        self.flank_threshold = flank_threshold
        self.cameras = cameras or []

    

    def draw_screen(self):
        map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        # scale = 1

        threshold = self.flank_threshold

        left_border = int(self.center - threshold * self.scale)
        right_border = int(self.center + threshold * self.scale)

        cv2.rectangle(map_img, (0, 0), (left_border, self.center), (50, 50, 100), -1)
        cv2.rectangle(map_img, (left_border, 0), (right_border, self.center), (50, 100, 50), -1)
        cv2.rectangle(map_img, (right_border, 0), (self.map_size, self.center), (100, 50, 50), -1)

        grid_step = self.flank_threshold

        for i in range(0, self.map_size, grid_step):
            cv2.line(map_img, (i, 0), (i, self.map_size), (40, 40, 40), 1)
            cv2.line(map_img, (0, i), (self.map_size, i), (40, 40, 40), 1)

        
        cv2.line(map_img, (0, self.center), (self.map_size, self.center), (80, 80, 80), 2)
        cv2.line(map_img, (self.center, 0), (self.center, self.map_size), (80, 80, 80), 2)

        for cam in self.cameras:
            cx = int(self.center + cam.global_X * self.scale)
            cy = int(self.center - cam.global_Z * self.scale)


            cv2.circle(map_img, (cx, cy), 6, (255, 255, 255), -1)
            cv2.putText(map_img, cam.camera_id, (cx + 8, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            yaw_rad = math.radians(cam.yaw_deg)
            arrow_len = 40
            ex = int(cx + arrow_len * math.sin(yaw_rad))
            ey = int(cy - arrow_len * math.cos(yaw_rad))
            cv2.arrowedLine(map_img, (cx, cy), (ex, ey), (0, 220, 255), 2, tipLength=0.3)

        return map_img

    
    def draw_paths(self, map_img, trails):
        for _, positions in trails.items():
            if len(positions) < 2:
                continue

            points = []
            for (X, Z) in positions:
                px = int(self.center + X * self.scale)
                py = int(self.center - Z * self.scale)
                points.append((px, py))

            n = len(points)
            for i in range(1, n):
                alpha = i / n
                intensity = int(80 + 175 * alpha)
                colour = (255, intensity, intensity)
                cv2.line(map_img, points[i - 1], points[i], colour, 2)

            for i, pt in enumerate(points[:-1]):
                alpha = i / n
                intensity = int(60 + 100 * alpha)
                cv2.circle(map_img, pt, 2, (255, intensity, intensity), -1)

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