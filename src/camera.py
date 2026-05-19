from src.pixel_to_world import Pixel2World
from src.distance import DistanceEstimator
import math


class CameraConfig:
    def __init__(self, camera_id, path, global_X, global_Z, yaw_deg, fov_horizontal, fov_vertical, cam_height, cam_pitch):
        self.camera_id = camera_id
        self.path = path
        self.global_X = global_X
        self.global_Z = global_Z

        self.yaw = math.radians(yaw_deg)

        self.pixel2world = Pixel2World(fov_horizontal, fov_vertical)
        self.distance_estimator = DistanceEstimator(fov_vertical, cam_height, cam_pitch)


    def estimate_distance(self, bbox, frame_height):
        return self.distance_estimator.estimate(bbox, frame_height)

    def pixel_to_global(self, cx, cy, frame_w, frame_h, distance):
        # get local coordinates in camera space
        X, Y, Z = self.pixel2world.calculate(cx, cy, frame_w, frame_h, distance)

        local_X = X
        local_Z = Z       

        global_X = (self.global_X + local_X * math.cos(self.yaw) - local_Z * math.sin(self.yaw))
        global_Z = (self.global_Z + local_X * math.sin(self.yaw) + local_Z * math.cos(self.yaw))

        return global_X, Y, global_Z 

       