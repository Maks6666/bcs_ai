import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch
from collections import defaultdict, deque
import numpy as np
import threading
import math
import json 
import socket

from conv_lstm_model import model

from decision_model.tactic_model import tactic_model
from weapon_model.weapon_model import weapon_model

import time
# from db import Table, session
import warnings


from src.threat import ThreatEstimator
from src.distance import DistanceEstimator
from src.counter import Counter
from src.items_encoder import ItemsEncoder
from src.tactic_predictor import TacticPredictor 
from src.weapon_counter import WeaponCounter
from src.command_predictor import CommandPredictor
from src.maneuver_predict import ManeuverPredictor

from src.map_window import MapWindow
from src.speed import Velocity
from src.intent import Inent
from src.priority import Priority
from src.gps import Local2GPS
from src.angle_calculator import AngleCalculator
from src.centers import Centers
from src.camera import CameraConfig, DynamicCameraConfig
from src.threat_field import ThreatField  
from src.threaded_stream_capture import ThreadedStreamCapture

from src.uwb_receiver import NoccelaPositionManager
from src.imu_reader import IMUReader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Tracker:
    def __init__(self, cameras, weapons, map_size, scale, max_dist,
                 lat, lon, heading, turret_position, raspberry_ip, raspberry_port):

        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        
        # self.tracker = DeepSort(max_age=5, max_iou_distance=0.4)
        self.cameras = cameras
        self.trackers = {camera.camera_id: DeepSort(max_age=5, max_iou_distance=0.4) for camera in cameras}

        
        self.model_link = "./yolo/main_weight.pt"
        self.model = self.load_model()
        self.yolo_names = self.model.names

        self.vehicles = {}
        self.weapons = weapons


        self.tactic_model = tactic_model
        self.maneuvers = ["Frontal attack", "Flank attack", "Outflank", "Mass attack", "Retreat", "Front-flank attack"]

        self.tactic_names = ['moving_back', 'center_flank', 'from_left_flank', 'from_right_flank']
        self.tactics = {}
        self.tactics_proba = {}

        self.threshold = 0.35
        self.frames = defaultdict(lambda: deque(maxlen=16))
        self.frames_length = 16
        self.frame_const = 80

        self.commands = ["ATGM", "Cluster shells", "Unitary shells", "FPV-drones", "Machine gun", "Rest of amunition"]

        self.last_updated = {}
        self.coordinates = {}

        self.last_frame = None
        self.logs = None
        self.map_frame = None 
        self.unique_logs = None 

        self.vehicle_real_width = {"TANK": 3.5, "IFV": 2.8, "APC": 2.5}
        self.distances = {}

        

        # camera angle - 90° - this is what camera sees
        # self.fov_horizontal = fov_horizontal
        # self.fov_vertical = fov_vertical
        # self.pixel2world = Pixel2World(self.fov_horizontal, self.fov_vertical)

        self.prev_positions = {}
        self.centers = {}
        self.positions = {}
        self.trails = defaultdict(lambda: deque(maxlen=60))

        self.max_dist = max_dist 

        self.threat = ThreatEstimator(self.yolo_names, self.max_dist)
        # self.distance = DistanceEstimator(self.yolo_names, self.vehicle_real_width)
        # self.vehicles_counter = VehiclesCounter()
        self.counter = Counter()
        self.items_encoder = ItemsEncoder(self.weapons)
        self.tactic_predictor = TacticPredictor(self.maneuvers)
        self.weapon_counter = WeaponCounter(self.weapons)
        self.command_predictor = CommandPredictor(self.commands)
        self.maneuver_predictor = ManeuverPredictor(self.tactics, self.tactics_proba, self.device)
        self.center = Centers()
        
        

        self.map_size = map_size
        self.scale = scale
        self.flank_threshold = 50
        self.map = MapWindow(self.map_size, self.scale, self.flank_threshold, self.cameras)

        self.flank_position = {'left_flank': [], 'center': [], 'right_flank': []}

        self.prev_time = {}       
        self.velocities = {}

        self.velocity_counter = Velocity(self.prev_positions, self.prev_time, self.velocities)

        # self.current_priority = None
        self.threat_scores = {}

        self.priority_calculator = Priority(self.threat_scores)

        self.intents_categories = ["attack", "retreat", "reposition", "idle"]
        self.history = defaultdict(lambda: deque(maxlen=30))
        self.intents = {}
        self.intent_predictor = Inent(self.history, self.intents)

        self.lat = lat
        self.lon = lon
        self.heading = heading
        self.gps_convertor = Local2GPS(self.lat, self.lon, self.heading)
        self.geo_positions = {}

        self.threat_field = ThreatField(self.map_size, self.scale, center=self.map_size // 2)

         # turret part 

        self.yaw_home = 90
        self.pitch_home = 60
        self.turret_position = turret_position

        self.angle = AngleCalculator(self.yaw_home, self.pitch_home)


        self.raspberry_ip = raspberry_ip
        self.raspberry_port = raspberry_port

        self.motion_threshold = 0.01

        # self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.sock.connect((self.raspberry_ip, self.raspberry_port))
        


    def load_model(self):
        model = YOLO(self.model_link)
        model.fuse()
        model.to(self.device)
        return model

    def results(self, frame):
        return self.model(frame, verbose=True)[0]
        # return self.model.predict(frame, verbose=False)

    def get_result(self, results, frame, camera_id):
        res_array = []
        bbox_conf_map = {}

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.threshold:
                bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                res_array.append((bbox, float(score), int(class_id)))

                cx = x1 + (x2 - x1) // 2
                cy = y1 + (y2 - y1) // 2
                bbox_conf_map[(cx, cy)] = score

        tracker = self.trackers[camera_id]
        tracks = tracker.update_tracks(raw_detections=res_array, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            bboxes = track.to_ltrb()
            idx = track.track_id
            class_id = track.get_det_class()

            cx = int((bboxes[0] + bboxes[2]) / 2)
            cy = int((bboxes[1] + bboxes[3]) / 2)


            # -----------------------------------------------------------------------------------------------------------
            closest_score = None
            min_dist = 50
            for (cx_det, cy_det), score in bbox_conf_map.items():
                # euclidean distance between centers BEFORE (cx_det, cy_det) and AFTER (cx, cy) DeepSort applying
                dist = np.hypot(cx - cx_det, cy - cy_det)
                if dist < min_dist:
                    min_dist = dist
                    closest_score = score

            score = closest_score if closest_score is not None else 0
            # score = ...
            # ----------------------------------------------------------------------------------------------------------------------------------

            results.append((bboxes, idx, class_id, round(score, 2)))

        return results

    def resize_frame(self, bboxes, h, w):
        x1, y1, x2, y2 = map(int, bboxes)

        x1 = max(0, x1-self.frame_const)
        y1 = max(0, y1-self.frame_const)

        x2 = min(w, x2+self.frame_const)
        y2 = min(h, y2+self.frame_const)

        return (x1, y1, x2, y2)

    # def input_weapons(self):
    #     while True:
    #         try:
    #             atgm = int(input("Input amount of ATGM: "))
    #             break
    #         except ValueError:
    #             print("Invalid input, please enter a number.")


    #     while True:
    #         try:
    #             cl_shells = int(input("Input amount of cluster shells: "))
    #             break
    #         except ValueError:
    #             print("Invalid input, please enter a number.")


    #     while True:
    #         try:
    #             u_shells = int(input("Input amount of unitary shells: "))
    #             break
    #         except ValueError:
    #             print("Invalid input, please enter a number.")


    #     while True:
    #         try:
    #             fpv = int(input("Input amount of FPV-drones: "))
    #             break
    #         except ValueError:
    #             print("Invalid input, please enter a number.")


    #     self.weapons["atgm"] = atgm
    #     self.weapons["cluster_shells"] = cl_shells
    #     self.weapons["unitary_shells"] = u_shells
    #     self.weapons["fpv_drones"] = fpv
    
    def predict_position(self, idx, t=2):
        if idx not in self.positions or idx not in self.velocities:
            return None 
        
        X, _, Z = self.positions[idx]
        v_x, v_z, _ = self.velocities[idx]

        X_f = X + v_x * t
        Z_f = Z + v_z * t

        return (X_f, Z_f)

    def draw_target(self, x_center, y_center, frame, colour):
        center = (x_center, y_center)
        cv2.circle(frame, center, 1, colour, -1)

    def draw(self, frame, resutls, priority):
        if resutls is not None:

            for (bboxes, idx, class_id, _) in resutls:
                x1, y1, x2, y2 = map(int, bboxes)
                score = self.threat_scores.get(idx, 0.0)

                colour = (0, 255, 0)
                if score is not None:
                    # print(score)
                    if score > 0.8:
                        colour = (0, 0, 255)
                    elif score > 0.5:
                        colour = (0, 165, 255)
                    else:
                        colour = (0, 255, 0)

                    if idx == priority:
                        c_x1, c_y1, c_x2, c_y2 = self.coordinates[idx]
                        x_center, y_center = self.center.get_center(c_x1, c_y1, c_x2, c_y2)
                        colour = (0, 0, 255)
                        self.draw_target(x_center, y_center, frame, colour)

                action = "..." if idx not in self.tactics else self.tactics[idx]
                action_proba = '...' if idx not in self.tactics_proba else round(self.tactics_proba[idx].item(), 2)*100
                dist = "..." if idx not in self.distances else self.distances[idx]

                position = self.positions[idx]
                X, Y, Z = position
                coord_text = f"({int(X)}m, {int(Y)}m, {int(Z)}m)"
                cv2.putText(frame, coord_text, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

            
                score = self.threat_scores[idx]

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                upper_text = f"{idx} | {self.yolo_names[int(class_id)]} | {action}: {action_proba}%"
                cv2.putText(frame, upper_text, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

                intent = self.intents.get(idx, None)
                lower_text = self.intents_categories[int(intent)] if intent is not None else '...'
                cv2.putText(frame, f"Threat: {score} | Inent: {lower_text}", (x1+50, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

            return frame

    def draw_total_coordinates(self, frame, results, h, w):
        if results is not None:

            x1_fin, y1_fin, x2_max, y2_max = 0, 0, 0, 0

            x1_s = []
            y1_s = []

            x2_s = []
            y2_s = []

            for (bboxes, _, _, _) in results:
                x1, y1, x2, y2 = map(int, bboxes)
                x1_s.append(x1)
                y1_s.append(y1)

                x2_s.append(x2)
                y2_s.append(y2)

                x1_min = min(x1_s)
                y1_min = min(y1_s)

                x2_max = max(x2_s)
                y2_max = max(y2_s)

                upd_bboxes = (x1_min, y1_min, x2_max, y2_max)
                x1_fin, y1_fin, x2_max, y2_max = self.resize_frame(upd_bboxes, h, w)

            cv2.rectangle(frame, (int(x1_fin), int(y1_fin)), (int(x2_max), int(y2_max)), (0, 0, 255), 2)

    def update_dict(self, resutls):
        current_idx = {idx for (_, idx, _, _) in resutls}
        for old_idx in list(self.tactics.keys()):
            if old_idx not in current_idx:
                del self.tactics[old_idx]
    
    def add_to_db(self):
        if len(self.vehicles) > 0 and len(self.tactics) > 0:
            for key, _ in self.vehicles.items():
                tactic = self.tactics.get(key)
                idx = key
                type = self.vehicles[key]

                existing = session.query(Table).filter_by(vehicle_index=idx).first()

                if not existing and tactic is not None:
                    # print(type, idx, tactic)
                    row = Table(type=type, vehicle_index=idx, action=tactic)
                    session.add(row)
            session.commit()


    def info_window(self, amount, amount_of_actions, tactical_maneuver, command, priority, prioriry_queue):
        window = np.zeros((400, 1050, 3), dtype=np.uint8)
        total_amount = len(self.vehicles)
        total_text = f"Total amount of detected objects: {total_amount}"
        cv2.putText(window, total_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        tanks, ifv, apc = amount
        amount_text = f"Amount of tanks: {tanks} | Amount of IFV: {ifv} | Amount of APC: {apc}"
        cv2.putText(window, amount_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(window, amount_of_actions, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        maneuver_text = f"Current enemy detected maneuver is: {tactical_maneuver}"
        cv2.putText(window, maneuver_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        command_text = f"Fire with: {command}"
        cv2.putText(window, command_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if priority is not None:
            priority_text = f"Priority target: {priority}"
        else:
            priority_text = f"Priority: has no priority target"
        cv2.putText(window, priority_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        l_f = len(self.flank_position['left_flank'])
        c_f = len(self.flank_position['center'])
        r_f = len(self.flank_position['right_flank'])

        flank_text = f"Targets on left flank: {l_f} | Targets on central flank: {c_f} | Targets on right flank: {r_f}"
        cv2.putText(window, flank_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        x_offset = 160
        for i, (k, v) in enumerate(prioriry_queue.items()):
            text = f"Place: {k} | Index: {v}"
            cv2.putText(window, text, (20 + i * x_offset, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)



        cv2.line(window, (0, 170), (1050, 170), (0, 255, 0), 1)
        cv2.putText(window, 'Unit information', (525, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        y_offset = 20
        if len(self.history) > 0:
            for i, (k, v) in enumerate(self.history.items()):
                v_type = v[-1]['v_type']
                X, Y, Z = v[-1]['pos']
                lat, lon = v[-1]['geo']
                _, _, speed = v[-1]['velocity']
                speed *= 3.6
                action = v[-1]['action']
                threat = v[-1]['threat']
                intent_idx = v[-1]['intent']
                intent = self.intents_categories[intent_idx] if intent_idx is not None else None
                
                text = f'IDX: {k}: {v_type} | {round(X, 2)}m/{round(Y, 2)}m/{round(Z, 2)}m | {round(speed, 2)}km/h | {action} | {threat} | {intent} | {lat}/{lon}°'
                cv2.putText(window, text, (20, 220 + y_offset * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return window

    def find_fusion_id(self, X, Z, class_id, threshold=15):
        for existing_id, (ex_X, _, ex_Z) in self.positions.items():
            existing_type = self.vehicles.get(existing_id)

            if existing_type != self.yolo_names[int(class_id)]:
                continue


            dist = math.sqrt((X - ex_X) ** 2 + (Z - ex_Z) ** 2)

            if dist < threshold:
                return existing_id
            
        return None
            
    def send_angles(self, yaw_target, pitch_target):
        message = {
            "yaw_target": float(yaw_target),
            "pitch_target": float(pitch_target)
        }

        file = (json.dumps(message) + '\n').encode('"utf-8"')
        print("[BCS] raw json:", file.strip())
        self.sock.sendall(file)


    def return_data(self, amount, actions, tactic_prediction, command, priority, prioriry_queue):
        total_amount = len(self.vehicles)

        tanks, ifv, apc = amount
        moving_forward, from_left_flank, from_right_flank, moving_back = actions

        l_f = len(self.flank_position['left_flank'])
        c_f = len(self.flank_position['center'])
        r_f = len(self.flank_position['right_flank'])


        data = {
            "total_amount": total_amount,
            "amount": {"tanks": tanks, "ifv": ifv, "apc": apc},
            "actions": {"moving_forward": moving_forward, "from_left_flank": from_left_flank, "from_right_flank": from_right_flank, "moving_back": moving_back},
            "tactic": tactic_prediction,
            "command": command,
            "priority": priority,
            "priorities": prioriry_queue,
            'flank': {'on_left_flank': l_f, 'on_central_flank': c_f, 'on_right_flank': r_f}
        }

        return data


    def return_unique_data(self):
        data_dict = {}
        if len(self.history) > 0:
            for k, v in self.history.items():
                object_dict = {}
                sub_dict = v[-1]
                # print(k, sub_dict)
                
                object_dict['v_type'] = sub_dict['v_type']
                object_dict['pos'] = sub_dict['pos']
                object_dict['geo'] = sub_dict['geo']

                _, _, speed = sub_dict['velocity']
                object_dict['speed'] = speed

                object_dict['distance'] = sub_dict['distance']
                object_dict['action'] = sub_dict['action']
                object_dict['threat'] = sub_dict['threat']
                object_dict['time'] = sub_dict['time']
                object_dict['intent'] = sub_dict['intent']

                data_dict[k] = object_dict

        
        return data_dict

    def __call__(self):
        # caps = {
        #     camera.camera_id: cv2.VideoCapture(camera.path)
        #     for camera in self.cameras
        # }

        caps = {}
        for camera in self.cameras:
            if isinstance(camera.path, str) and camera.path.startswith("http"):
                caps[camera.camera_id] = ThreadedStreamCapture(camera.path)
            else:
                caps[camera.camera_id] = cv2.VideoCapture(camera.path)

        for cap in caps.values():
            assert cap.isOpened()

        subtractors = {
            camera.camera_id: cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=40, detectShadows=False
            )
            for camera in self.cameras
        }

        while True:
            self.positions = {}
            self.threat_scores.clear()
            self.flank_position = {'left_flank': [], 'center': [], 'right_flank': []}

            current_ids_per_camera = {camera.camera_id: set() for camera in self.cameras}

            current_ids = set()
            any_frame = False

            camera_frames = {}
            camera_results = {}
            camera_sizes = {}

            for camera in self.cameras:
                cap = caps[camera.camera_id]

                ret, frame = cap.read()
                if not ret:
                    continue

                any_frame = True
                h, w, _ = frame.shape

                # == MOTION DETECTION ==

                mask = subtractors[camera.camera_id].apply(frame)
                if np.count_nonzero(mask) / mask.size < self.motion_threshold:
                    continue

                # --------------------------------------

                results = self.results(frame)
                resutls_array = self.get_result(results, frame, camera.camera_id)
                # current_ids = {idx for (_, idx, _, _) in resutls_array}

                global_results_array = []

                for (bboxes, idx, class_id, conf) in resutls_array:
                    global_id = f"{camera.camera_id}_{idx}"
    
                    upd_bboxes = self.resize_frame(bboxes, h, w)
                    x1, y1, x2, y2 = map(int, upd_bboxes)
                   
                    # D = self.distance.estimate(bboxes, idx, class_id, w)
                    D = camera.estimate_distance(bboxes, h)

                    x1_, y1_, x2_, y2_ = map(int, bboxes)
                    cx, cy = self.center.get_center(x1_, y1_, x2_, y2_)

                    X, Y, Z = camera.pixel_to_global(cx, cy, w, h, D)

                    # print(self.positions)

                    # -----------------------------------------------------------------------------------------------------------

                    # EXAMPLE 

                    # cam_1
                    # X, Z = 100, 250
                    # self.positions = {}  # empty at the beginning
                    # find_fusion_id(100, 250) → None
                    # self.positions["cam_1_1"] = (100, 0, 250)

                    # # cam_2
                    # X, Z = 102, 248
                    # self.positions = {
                    #     "cam_1_1": (100, 0, 250)
                    # }

                    # find_fusion_id(102, 248) → compare with cam_1_1 (100, 250) -> same object

                    # --------------------------------------------------------------------------

                    fusion_id = self.find_fusion_id(X, Z, class_id)
                    if fusion_id is None:
                        object_id = global_id
                    else:
                        object_id = fusion_id

                    # current_ids.add(object_id)

                    current_ids_per_camera[camera.camera_id].add(object_id)

                    # -----------------------------------------------------------------------------------------------------------

                    self.positions[object_id] = (X, Y, Z)
                    self.trails[object_id].append((X, Z))
                    self.vehicles[object_id] = self.yolo_names[int(class_id)]
                    self.coordinates[object_id] = (x1, y1, x2, y2)
                    self.distances[object_id] = D
                    self.centers[object_id] = (cx, cy)

                    v_type = self.vehicles[object_id]

                    # print(self.positions) -> X, Y, Z are separate for each camera and separate for each object, because of global_id

                    lat, lon = self.gps_convertor.convert(X, Z)
                    self.geo_positions[global_id] = (lat, lon)

                    # ----------------------------------------------------------------------------------------------------------------------   
                    # THREAT ESTIMATION
                    # ---------------------------------------------------------------------------------------------------------------------- 


                    action = self.tactics.get(object_id, "Analyzing...")
                    action_proba = self.tactics_proba.get(object_id, None)

                    curr_pos = (X, Z)
                    prev_pos = self.prev_positions.get(object_id)
                    future_pos = self.predict_position(object_id)

                    intent = self.intent_predictor.calculate(object_id)

                    score = self.threat.score(class_id, D, action, action_proba, conf, curr_pos, prev_pos, future_pos, intent)

                    self.threat_scores[object_id] = score

                    # ---------------------------------------------------------------------------------------------------------------------- 

                    v_x, v_z, speed = self.velocity_counter.calculate(object_id, X, Z)

                    self.history[object_id].append({
                        "v_type": v_type,
                        "pos": (X, Y, Z),
                        "geo": (lat, lon),
                        "velocity": (v_x, v_z, speed),
                        "distance": D,
                        "action": action,
                        "threat": score,
                        "time": time.time(),
                        "intent": intent,
                        "camera_id": camera.camera_id
                    })

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    crop = cv2.resize(crop, (128, 128))
                    self.frames[object_id].append(crop)

                    if len(self.frames[object_id]) == self.frames_length:
                        frames = list(self.frames[object_id])
                        threading.Thread(target=self.maneuver_predictor.prediction, args=(frames, object_id)).start()
                        self.frames[object_id].clear()

                    global_results_array.append((bboxes, object_id, class_id, conf))


                camera_frames[camera.camera_id] = frame
                camera_results[camera.camera_id] = global_results_array
                camera_sizes[camera.camera_id] = (h, w)

            if not any_frame:
                break

            current_ids = set().union(*current_ids_per_camera.values())

            for idx_ in list(self.history.keys()):
                if idx_ not in current_ids:
                    del self.history[idx_]
                    self.intents.pop(idx_, None)
                    self.positions.pop(idx_, None)
                    self.velocities.pop(idx_, None)
                    self.distances.pop(idx_, None)
                    self.frames.pop(idx_, None)
                    self.vehicles.pop(idx_, None)
                    self.coordinates.pop(idx_, None)
                    self.tactics.pop(idx_, None)
                    self.tactics_proba.pop(idx_, None)
                    self.geo_positions.pop(idx_, None)
                    self.trails.pop(idx_, None)

            tank, ifv, apc = self.counter.count_vehicles(self.vehicles)
            amount = (tank, ifv, apc)

            array = self.items_encoder.encode(tank, ifv, apc)

            command = self.command_predictor.predict_command(array, self.vehicles)
            if command:
                self.weapon_counter.fire(command=command)


            amount_of_actions, actions = self.counter.count_statuses(self.tactics)
            tactic_prediction = self.tactic_predictor.predict_tactic(actions, self.tactics)

            priority = self.priority_calculator.choose_target()

            if priority and priority in self.positions:
                X, Y, Z = self.positions[priority]
                Y = 0.0

                turret_x, turret_y, turret_z = self.turret_position

                yaw_target, pitch_target = self.angle.calculate_absolute(turret_x, turret_y, turret_z, X, Y, Z)

                print(f"[BCS] send -> target={priority}, yaw={yaw_target:.2f}, pitch={pitch_target:.2f}")
                # self.send_angles(yaw_target, pitch_target)

            priority_queue = self.priority_calculator.priority_list(priority)

            # === MAP DRAWING ===

            map_img = self.map.draw_screen()
            map_img = self.map.draw_paths(map_img, self.trails)

            self.threat_field.update(self.positions, self.threat_scores)
            map_img = self.threat_field.draw(map_img)


            map_img_ = self.map.draw_objects(map_img, self.vehicles, self.positions, self.threat_scores, priority)

            self.counter.count_flanks(self.positions, self.scale, self.map_size, self.flank_threshold, self.flank_position)

            for camera_id, frame in camera_frames.items():
                resutls_array = camera_results[camera_id]
                h, w = camera_sizes[camera_id]

                frame = self.draw(frame, resutls_array, priority)
                self.draw_total_coordinates(frame, resutls_array, h, w)

                cv2.imshow(f"YOLO Tracker | {camera_id}", frame)

                self.last_frame = frame

            info_window = self.info_window(amount, amount_of_actions, tactic_prediction, command, priority, priority_queue)

            data = self.return_data(amount, actions, tactic_prediction, command, priority, priority_queue)
            unique_data = self.return_unique_data()

            self.map_frame = map_img_
            self.logs = data
            self.unique_logs = unique_data

            cv2.imshow("Top-Down Map", map_img_)
            cv2.imshow("info_window", info_window)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        for cap in caps.values():
            cap.release()

        cv2.destroyAllWindows()




weapons = {'atgm': 30, 'cluster_shells': 30, 'unitary_shells': 30, 'fpv_drones': 30}
# path = "./video/test_video_1.mp4"
# path = './video/test_video_1.mp4'


# using MPU to calculate yaw_deg and cam_pitch

CAMERA_1_IP = "172.20.10.4"  
STREAM_URL_1 = f"http://{CAMERA_1_IP}:81/stream"

CAMERA_2_IP = "192.168.1.142"  
STREAM_URL_2 = f"http://{CAMERA_2_IP}:81/stream"



# (!!!!!) Change camera IDs to 2435000088 and 2435000119

cameras = [
    CameraConfig(
        # camera_id="cam_1",
        camera_id = "2435000088",
        # path="./video/test_video_1.mp4", 
        path = STREAM_URL_1,
        global_X=0,
        global_Z=0,
        yaw_deg=-5,
        fov_horizontal=73.7,
        fov_vertical=46.5,
        cam_height=15,   
        cam_pitch=10,
    ),
    # CameraConfig(
    #     # camera_id="cam_2",
    #     camera_id = "2435000119",
    #     # path = STREAM_URL_2,
    #     path="./video/test_video_1.mp4", 
    #     global_X=5,
    #     global_Z=5,
    #     yaw_deg=0,
    #     fov_horizontal=73.7,
    #     fov_vertical=46.5,
    #     cam_height=15,   
    #     cam_pitch=10,
    # ),
]



# cameras = [
#     DynamicCameraConfig(
#         camera_id="T0",        # ← совпадает с tag_id в UWB теге
#         path="http://192.168.1.104:81/stream",  # ← стрим с ESP32-CAM
#         fov_horizontal=73.7,
#         fov_vertical=46.5,
#     ),
#     DynamicCameraConfig(
#         camera_id="T1",
#         path="http://192.168.1.105:81/stream",
#         fov_horizontal=73.7,
#         fov_vertical=46.5,
#     ),
# ]


# ANCHORS = {
#     "A0": {"host": "192.168.1.101", "port": 8080,
#            "X": 0.0,  "Y": 0.0, "Z": 0.0},
#     "A1": {"host": "192.168.1.102", "port": 8080,
#            "X": 40.0, "Y": 0.5, "Z": 0.0},
#     "A2": {"host": "192.168.1.103", "port": 8080,
#            "X": 20.0, "Y": 1.5, "Z": 35.0},
# }

# uwb = UWBPositionManager(anchors=ANCHORS, cameras=cameras)
# uwb.start()





imu_readers = [
    IMUReader(camera=cameras[0], host="172.20.10.4", pitch_offset=-4.11, yaw_offset=0, gyro_z_sign=-1),
    # IMUReader(camera=cameras[1], host="192.168.1.142", pitch_offset=-4.11, yaw_offset=0, gyro_z_sign=-1),  
]
for imu in imu_readers:
    imu.start()


manager = NoccelaPositionManager(
    url = "ws://172.20.10.3:3000/realtime",
    # url="ws://localhost:3000/realtime",
    cameras=cameras
)
manager.start()




map_size = 600
scale = 1.0
max_dist = 1000


# coordinates 
lat = 50.5724
lon = 31.4883

# camera heading
heading = 20


# turret_position = (0.3, 0.0, 0.1)
# turret_position = (-30, 0.0, 0.0)
turret_position = (0.0, 0.0, 0.0)

# camera fov
# hfov = 73.7
# vfov = 46.5


raspberry_ip = "192.168.1.141"
raspberry_port = 5000

tracker = Tracker(cameras, weapons, map_size, scale, max_dist,
                  lat, lon, heading, turret_position, raspberry_ip, raspberry_port)
tracker()

# python3 tracker.py