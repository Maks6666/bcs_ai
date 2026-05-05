import math 


class Pixel2World:
    def __init__(self, fov_horizontal, fov_vertical):
        self.fov_horizontal = fov_horizontal
        self.fov_vertical = fov_vertical
    
    def calculate(self, cx, cy, frame_w, frame_h, distance):
        center_x = frame_w / 2
        center_y = frame_h / 2

        hfov = math.radians(self.fov_horizontal)
        vfov = math.radians(self.fov_vertical)

        fx = frame_w / (2 * math.tan(hfov / 2))
        fy = frame_h / (2 * math.tan(vfov / 2))

        X = distance * (cx - center_x) / fx
        Y = distance * (cy - center_y) / fy
        Z = distance

        return X, Y, Z