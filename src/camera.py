from src.pixel_to_world import Pixel2World
from src.distance import DistanceEstimator
import math
import threading
import requests
import cv2
import numpy as np 



class DynamicCameraConfig:
    def __init__(self, camera_id, path, global_X=0, global_Z=0,
                 yaw_deg=0, fov_horizontal=73.7, fov_vertical=46.5,
                 cam_height=0, cam_pitch=0):
        self.camera_id    = camera_id
        self.path         = path
        self.global_X     = global_X
        self.global_Z     = global_Z
        self.yaw_deg      = yaw_deg
        self.cam_pitch    = cam_pitch
        self.cam_height   = cam_height
        self.fov_vertical = fov_vertical
        self._lock        = threading.Lock()

        self.pixel2world  = Pixel2World(fov_horizontal, fov_vertical)

    def estimate_distance(self, bbox, frame_height):
        with self._lock:
            pitch  = self.cam_pitch
            height = self.cam_height
       
        estimator = DistanceEstimator(self.fov_vertical, height, pitch)
        return estimator.estimate(bbox, frame_height)

    def pixel_to_global(self, cx, cy, frame_w, frame_h, distance):
        with self._lock:
            yaw_rad = math.radians(self.yaw_deg)
            cam_X   = self.global_X
            cam_Z   = self.global_Z

        X, Y, Z = self.pixel2world.calculate(cx, cy, frame_w, frame_h, distance)

        global_X = cam_X + X * math.cos(yaw_rad) - Z * math.sin(yaw_rad)
        global_Z = cam_Z + X * math.sin(yaw_rad) + Z * math.cos(yaw_rad)

        return global_X, Y, global_Z

    def update_from_uwb(self, X: float, Z: float, height: float):
        with self._lock:
            self.global_X   = X
            self.global_Z   = Z
            self.cam_height = height

    def update_from_imu(self, yaw: float, pitch: float):
        with self._lock:
            self.yaw_deg   = yaw
            self.cam_pitch = pitch



class CameraConfig:
    def __init__(self, camera_id, path, global_X, global_Z, yaw_deg,
                 fov_horizontal, fov_vertical, cam_height, cam_pitch):
        
        self.camera_id = camera_id
        self.path = path
        self.global_X = global_X
        self.global_Z = global_Z

        self.yaw = math.radians(yaw_deg)
        self.yaw_deg = yaw_deg
        self.cam_pitch = cam_pitch
        self.cam_height = cam_height

        self._lock = threading.Lock()

        self.pixel2world = Pixel2World(fov_horizontal, fov_vertical)
        self.distance_estimator = DistanceEstimator(fov_vertical, cam_height, cam_pitch)

    
    def get_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def estimate_distance(self, bbox, frame_height):
        return self.distance_estimator.estimate(bbox, frame_height)

    def pixel_to_global(self, cx, cy, frame_w, frame_h, distance):
        X, Y, Z = self.pixel2world.calculate(cx, cy, frame_w, frame_h, distance)

        local_X = X
        local_Z = Z

        global_X = (self.global_X + local_X * math.cos(self.yaw) - local_Z * math.sin(self.yaw))
        global_Z = (self.global_Z + local_X * math.sin(self.yaw) + local_Z * math.cos(self.yaw))

        return global_X, Y, global_Z

    def update_from_uwb(self, X: float, Z: float, height: float):
        with self._lock:
            self.global_X = X
            self.global_Z = Z
            self.cam_height = height

    def update_from_imu(self, yaw: float, pitch: float):
        with self._lock:
            print(f"[NEW YAW] -> {yaw}; [NEW PITCH] -> {pitch}")
            self.yaw_deg = yaw
            self.yaw = math.radians(yaw)
            self.cam_pitch = pitch







# class CameraConfig:
#     def __init__(self, camera_id, path, global_X, global_Z, yaw_deg, fov_horizontal, fov_vertical, cam_height, cam_pitch):
#         self.camera_id = camera_id
#         self.path = path
#         self.global_X = global_X
#         self.global_Z = global_Z

#         self.yaw = math.radians(yaw_deg)

#         self.pixel2world = Pixel2World(fov_horizontal, fov_vertical)
#         self.distance_estimator = DistanceEstimator(fov_vertical, cam_height, cam_pitch)


#     def estimate_distance(self, bbox, frame_height):
#         return self.distance_estimator.estimate(bbox, frame_height)

#     def pixel_to_global(self, cx, cy, frame_w, frame_h, distance):
#         # get local coordinates in camera space
#         X, Y, Z = self.pixel2world.calculate(cx, cy, frame_w, frame_h, distance)

#         local_X = X
#         local_Z = Z       

#         global_X = (self.global_X + local_X * math.cos(self.yaw) - local_Z * math.sin(self.yaw))
#         global_Z = (self.global_Z + local_X * math.sin(self.yaw) + local_Z * math.cos(self.yaw))

#         return global_X, Y, global_Z 
    

#     def update_from_uwb(self, X: float, Z: float, height: float):
#         with self._lock:
#             self.global_X   = X
#             self.global_Z   = Z
#             self.cam_height = height

#     def update_from_imu(self, yaw: float, pitch: float):
#         with self._lock:
#             self.yaw_deg   = yaw
#             self.cam_pitch = pitch

       


# class DynamicCameraConfig:
#     def __init__(
#         self,
#         camera_id,
#         path,
#         fov_horizontal,
#         fov_vertical,
#         yaw_deg=0.0,
#         cam_pitch=0.0,
#         global_X=0.0,
#         global_Z=0.0,
#         cam_height=0.0,
#     ):
#         self.camera_id = camera_id
#         self.path = path

#         self.fov_horizontal = float(fov_horizontal)
#         self.fov_vertical = float(fov_vertical)

#         # Положение камеры, обновляется от UWB
#         self.global_X = float(global_X)
#         self.global_Z = float(global_Z)
#         self.cam_height = float(cam_height)

#         # Ориентация камеры, обновляется от IMU
#         self.yaw_deg = float(yaw_deg)
#         self.cam_pitch = float(cam_pitch)

#         self.pixel2world = Pixel2World(
#             self.fov_horizontal,
#             self.fov_vertical
#         )

#         self._lock = threading.Lock()

#     def update_from_uwb(
#         self,
#         X: float,
#         Z: float,
#         height: float
#     ) -> None:
#         with self._lock:
#             self.global_X = float(X)
#             self.global_Z = float(Z)
#             self.cam_height = float(height)

#     def update_from_imu(
#         self,
#         yaw: float,
#         pitch: float
#     ) -> None:
#         with self._lock:
#             self.yaw_deg = float(yaw)
#             self.cam_pitch = float(pitch)

#     def estimate_distance(
#         self,
#         bbox,
#         frame_height: int
#     ) -> float:
#         # Берём актуальные значения на момент вычисления
#         with self._lock:
#             cam_height = self.cam_height
#             cam_pitch = self.cam_pitch

#         # Создаём estimator с текущим pitch
#         distance_estimator = DistanceEstimator(
#             self.fov_vertical,
#             cam_height,
#             cam_pitch
#         )

#         return distance_estimator.estimate(
#             bbox,
#             frame_height
#         )

#     def pixel_to_global(
#         self,
#         cx: float,
#         cy: float,
#         frame_w: int,
#         frame_h: int,
#         distance: float
#     ):
#         # Сначала считаем положение объекта относительно камеры
#         local_X, local_Y, local_Z = self.pixel2world.calculate(
#             cx,
#             cy,
#             frame_w,
#             frame_h,
#             distance
#         )

#         # Получаем согласованный снимок текущего состояния камеры
#         with self._lock:
#             camera_X = self.global_X
#             camera_Z = self.global_Z
#             yaw_rad = math.radians(self.yaw_deg)

#         # Поворот локальных координат относительно yaw камеры
#         global_X = (
#             camera_X
#             + local_X * math.cos(yaw_rad)
#             - local_Z * math.sin(yaw_rad)
#         )

#         global_Z = (
#             camera_Z
#             + local_X * math.sin(yaw_rad)
#             + local_Z * math.cos(yaw_rad)
#         )

#         return global_X, local_Y, global_Z