import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch
from collections import defaultdict, deque
import numpy as np
import threading

from models.conv_lstm_model import model

from decision_model.tactic_model import tactic_model
from weapon_model.weapon_model import weapon_model

import time
from db import Table, session
import warnings


from src.threat import ThreatEstimator
from src.distance import DistanceEstimator
from src.vehicles_counter import VehiclesCounter
from src.status_counter import StatusCounter
from src.items_encoder import ItemsEncoder
from src.tactic_predictor import TacticPredictor 
from src.weapon_counter import WeaponCounter
from src.command_predictor import CommandPredictor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Tracker:
    def __init__(self, path, weapons):
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tracker = DeepSort(max_age=5, max_iou_distance=0.4)
        self.path = path
        self.model_link = "/Users/maxkucher/PycharmProjects/bcs_ai_/yolo/best_v02.pt"
        self.model = self.load_model()
        self.yolo_names = self.model.names

        self.vehicles = {}
        self.weapons = weapons


        self.tactic_model = tactic_model
        self.maneuvers = ["Frontal attack", "Flank attack", "Outflank", "Mass attack", "Retreat", "Front-flank attack"]

        self.tactic_names = ['moving_back', 'center_flank', 'from_left_flank', 'from_right_flank']
        self.tactics = {}

        self.threshold = 0.35
        self.frames = defaultdict(lambda: deque(maxlen=16))
        self.frames_length = 16
        self.frame_const = 80

        self.commands = ["ATGM", "Cluster shells", "Unitary shells", "FPV-drones", "Machine gun", "Rest of amunition"]

        self.last_updated = {}
        self.coordinates = {}

        self.last_frame = None
        self.logs = None

        self.vehicle_real_width = {"TANK": 3.5, "IFV": 2.8, "APC": 2.5}
        self.distances = {}

        self.threat_scores = {}

        self.threat = ThreatEstimator(self.yolo_names)
        self.distance = DistanceEstimator(self.yolo_names, self.vehicle_real_width)
        self.counter = VehiclesCounter()
        self.status_counter = StatusCounter()
        self.items_encoder = ItemsEncoder(self.weapons)
        self.tactic_predictor = TacticPredictor(self.maneuvers)
        self.weapon_counter = WeaponCounter(self.weapons)
        self.command_predictor = CommandPredictor(self.commands)


    def load_model(self):
        model = YOLO(self.model_link)
        model.fuse()
        model.to(self.device)
        return model

    def results(self, frame):
        return self.model(frame, verbose=True)[0]
        # return self.model.predict(frame, verbose=False)

    def get_result(self, results, frame):
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

        tracks = self.tracker.update_tracks(raw_detections=res_array, frame=frame)

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

    def prediction(self, frames, idx) -> None:
        clip = np.stack(frames, axis=0)
        clip = np.transpose(clip, (0, 3, 1, 2))
        clip = torch.tensor(clip, dtype=torch.float32).unsqueeze(0) / 255.0


        clip = clip.permute(0, 2, 1, 3, 4)
        # print(clip.shape)
        res = model.predict(clip)
        name = self.tactic_names[int(res)]
        self.tactics[idx] = str(name)

        # print(self.tactics)

    def get_center(self, x1, y1, x2, y2):
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        return x_center, y_center

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
                        x_center, y_center = self.get_center(c_x1, c_y1, c_x2, c_y2)
                        colour = (0, 0, 255)
                        self.draw_target(x_center, y_center, frame, colour)

                action = "Analyzing..." if idx not in self.tactics else self.tactics[idx]
                dist = "Calculating..." if idx not in self.distances else self.distances[idx]

                text = f"{idx} | {self.yolo_names[int(class_id)]} | {dist}m | {action}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

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


    def predict_command(self, array):
        if len(self.vehicles) > 0:
            res = weapon_model.predict(array)
            command = self.commands[int(res)]
            return command
        else:
            return None

    

    def choose_target(self):
        if not self.threat_scores:
            return None
        
        priority = max(self.threat_scores, key=self.threat_scores.get)
        return priority


    
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

    def info_window(self, amount, amount_of_actions, tactical_maneuver, command, priority):
        window = np.zeros((400, 850, 3), dtype=np.uint8)
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

        return window

    def return_data(self, amount, actions, tactic_prediction, command, priority):
        total_amount = len(self.vehicles)

        tanks, ifv, apc = amount
        moving_forward, from_left_flank, from_right_flank, moving_back = actions
        data = {
            "total_amount": total_amount,
            "amount": {"tanks": tanks, "ifv": ifv, "apc": apc},
            "actions": {"moving_forward": moving_forward, "from_left_flank": from_left_flank, "from_right_flank": from_right_flank, "moving_back": moving_back},
            "tactic": tactic_prediction,
            "command": command,
            "priority": priority
        }

        return data

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        # self.input_weapons()
        # print(self.weapons)

        while True:
            self.vehicles = {}
            ret, frame = cap.read()

            if not ret:
                break

            if frame is not None:
                h, w, _ = frame.shape


            results = self.results(frame)
            resutls_array = self.get_result(results, frame)

            for (bboxes, idx, class_id, conf) in resutls_array:
                self.vehicles[idx] = self.yolo_names[int(class_id)]

                upd_bboxes = self.resize_frame(bboxes, h, w)
                x1, y1, x2, y2 = map(int, upd_bboxes)
                self.coordinates[idx] = (x1, y1, x2, y2)

                # self.estimate_distance(bboxes, idx, class_id)
                D = self.distance.estimate(bboxes, idx, class_id)
                self.distances[idx] = D

                D = self.distances[idx]
                action = self.tactics.get(idx, "Analyzing...")
                
                score = self.threat.score(class_id, D, action, conf)
                # print(score)

                self.threat_scores[idx] = score


                crop = frame[y1:y2, x1:x2]
                if crop.size is None:
                    continue

                crop = cv2.resize(crop, (128, 128))
                self.frames[idx].append(crop)

                if len(self.frames[idx]) == self.frames_length:
                    frames = list(self.frames[idx])
                    threading.Thread(target=self.prediction, args=(frames, idx)).start()
                    self.frames[idx].clear()

            
            tank, ifv, apc = self.counter.count(self.vehicles)
            amount = (tank, ifv, apc)

            array = self.items_encoder.encode(tank, ifv, apc)
        
            # command = self.predict_command(array)

            command = self.command_predictor.predict_command(array, self.vehicles)
            if command:
                self.weapon_counter.fire(command=command)
                # print(self.weapons)

            self.update_dict(resutls_array)

            amount_of_actions, actions = self.status_counter.count_statuses(self.tactics)

            tactic_prediction = self.tactic_predictor.predict_tactic(actions, self.tactics)

            priority = self.choose_target()

            frame = self.draw(frame, resutls_array, priority)
            self.draw_total_coordinates(frame, resutls_array, h, w)

            info_window = self.info_window(amount, amount_of_actions, tactic_prediction, command, priority)

            self.add_to_db()

            data = self.return_data(amount, actions, tactic_prediction, command, priority)
            # print(data)

            self.last_frame = frame
            self.logs = data

            cv2.imshow('YOLO Tracker', frame)
            cv2.imshow('info_window', info_window)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # return frame (bytecode: jpeg) + logs
        cap.release()
        cv2.destroyAllWindows()


weapons = {'atgm': 30, 'cluster_shells': 30, 'unitary_shells': 30, 'fpv_drones': 30}
path = "./video/test_video_1.mp4"
tracker = Tracker(path, weapons)
tracker()
