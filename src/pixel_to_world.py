import math 


class Pixel2World:
    def __init__(self, fov_horizontal):
        self.fov_horizontal = fov_horizontal
    
    def calculcate(self, cx, frame_w, distance):
        center_x = frame_w / 2

        dx = (cx - center_x) / center_x
        # dy = (cy - center_y) / center_y

        # dx / dy - normalized displacement of an object from the center of the frame
        # dx є [-1, 1]

        # Convert normalized horizontal offset (dx in [-1, 1]) into a real-world viewing angle:
        # camera sees from -FOV/2 to +FOV/2, so dx maps linearly to that range (0 = straight ahead),
        # then convert degrees to radians for trigonometric functions

        # math.pi / 180 - turn angle into radians 

        angle_x = dx * (self.fov_horizontal / 2) * math.pi / 180

        X = distance * math.tan(angle_x)
        Y = distance

        return X, Y 