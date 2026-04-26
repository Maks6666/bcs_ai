import math 

class AngleCalculator:
    def __init__(self, hfov, vfov, yaw_home, pitch_home):
        self.hfov = hfov
        self.vfov = vfov

        self.yaw_home = yaw_home
        self.pitch_home = pitch_home

    def deadzone(self, value, threshold=1.0):
        return 0 if abs(value) < threshold else value
    
    def calculate(self, cx, cy, frame_width, frame_height):
        center_x = frame_width / 2
        center_y = frame_height / 2

        dx = cx - center_x
        dy = cy - center_y

        yaw_delta = dx * (self.hfov / frame_width)
        pitch_delta = -dy * (self.vfov / frame_height)

        yaw_delta = self.deadzone(yaw_delta, 0.5)
        pitch_delta = self.deadzone(pitch_delta, 0.5)
        
        yaw_delta *= 0.3
        pitch_delta *= 0.3

        return yaw_delta, pitch_delta
    
    def calculate_absolute(self, turret_x, turret_y, turret_z, X, Y, Z):
        dx = X - turret_x
        dy = Y - turret_y
        dz = Z - turret_z

        yaw = math.degrees(math.atan2(dx, dz))
        pitch = math.degrees(math.atan2(dy, math.sqrt(dx * dx + dz * dz)))

        yaw_target = self.yaw_home - yaw
        pitch_target = self.pitch_home - pitch

        return yaw_target, pitch_target
