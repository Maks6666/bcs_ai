import math 

class PixelToWorld:
    def __init__(self, fov_horizontal, fov_vertical):
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
    
    def calculate(self, bbox, frame_width, frame_height, D):
        x1, y1, x2, y2 = bbox

        # Calculate the center of the bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        hfov = math.radians(self.fov_horizontal)
        vfov = math.radians(self.fov_vertical)

        fx = frame_width / (2 * math.tan(hfov / 2))
        fy = frame_height / (2 * math.tan(vfov / 2))


        X = (center_x - frame_width / 2) * D / fx
        Y = (center_y - frame_height / 2) * D / fy
        Z = D

        return round(X, 2), round(Y, 2), round(Z, 2)