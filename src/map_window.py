import cv2
import numpy as np

class MapWindow:
    def __init__(self, map_size, scale):
        self.map_size = map_size
        self.center = map_size // 2
        self.scale = scale

    def draw_screen(self):
        map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        grid_size = 50

        for x in range(0, self.map_size, grid_size):
            cv2.line(map_img, (x, 0), (x, self.map_size), (50, 50, 50), 1)
            cv2.line(map_img, (0, x), (self.map_size, x), (50, 50, 50), 1)


        cv2.line(map_img, (self.center, 0), (self.center, self.map_size), (80, 80, 80), 2)
        cv2.line(map_img, (0, self.center), (self.map_size, self.center), (80, 80, 80), 2)

        cv2.circle(map_img, (self.center, self.center), 6, (255, 255, 255), -1)
        cv2.putText(map_img, "CAMERA", (self.center + 5, self.center - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return map_img
    

    def draw_objects(self, positions, map_img):
        for idx, (X, _, Z) in positions.items():
            px = int(self.center + X * self.scale)
            pz = int(self.center - Z * self.scale)


            cv2.circle(map_img, (px, pz), 5, (60, 60, 60), -1)
            cv2.putText(map_img, idx, (px, pz),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
        
        return map_img

        