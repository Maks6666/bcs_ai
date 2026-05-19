import math

# class DistanceEstimator:
#     def __init__(self, yolo_names: list, vehicle_real_width: dict):
#         self.yolo_names = yolo_names
#         self.vehicle_real_width = vehicle_real_width
#         # self.distances = distances

#         self.f_mm = 8.0
#         self.sensor_width = 6.4
    
#     def estimate(self, bbox, idx, class_id, frame_width):
#         x1, y1, x2, y2 = map(int, bbox)
#         name = self.yolo_names[int(class_id)]

#         P = x2 - x1
#         if P <= 0:
#             return None

#         W = self.vehicle_real_width.get(name, 3.0)

#         f_mm = self.f_mm
#         sensor_width = self.sensor_width

#         f = (f_mm * frame_width) / sensor_width

#         D = (W * f) / P

#         return round(D, 2)

    
class DistanceEstimator:
    def __init__(self, v_fov, cam_height, cam_pitch):
        self.v_fov = math.radians(v_fov)
        self.cam_height = cam_height
        self.cam_pitch = math.radians(cam_pitch)

    def estimate(self, bbox, frame_height):
        _, _, _, y2 = bbox

        y_wheel = y2

        alpha = ((y_wheel - frame_height / 2) / frame_height) * self.v_fov

        beta = alpha + self.cam_pitch

        if beta <= 0:
            return None
        
        D = self.cam_height / math.tan(beta)

        return round(D, 2)

