import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch 
import numpy as np 
from sklearn.cluster import DBSCAN
import threading

from collections import defaultdict, deque

from src.optical_flow import OpticalFlow
from src.focal_length import FocalLength
from src.threat_estimation import ThreatEstimator
from src.group_tracking import GroupTracker

from infantry_tactics_model.infantry_model import model




class Tracker:
    def __init__(self, path, device, yolo_link):
        self.device = device
        self.path = path 
        self.yolo_link = yolo_link
        self.tracker = DeepSort(max_age=5, max_iou_distance=0.4)
        self.model = self.load_model()
        self.names = self.model.names

        self.threshold = 0.4
        self.group_box_const = 15   

        self.optical_flow = OpticalFlow()

        self.group_buffers = defaultdict(lambda: deque(maxlen=16))
        self.group_actions = {}
        self.action_proba = {}

        self.prev_groups = {}
        self.next_group_id = 0
        self.group_tracker = GroupTracker(self.prev_groups, self.next_group_id)

        self.frames_needed = 16
        self.tactics = ['advance', 'disperse', 'halt', 'regroup', 'retreat']

        self.focal_length = FocalLength()
        self.group_dist = {}

        self.action_threats = {
                    "advance": 1.0,
                    "regroup": 0.7,
                    "disperse": 0.5,
                    "halt": 0.2,
                    "retreat": 0.1
                    }
        
        self.threat_estimator = ThreatEstimator(self.action_threats)
        self.group_threat = {}




    def load_model(self):
        model = YOLO(self.yolo_link)
        model.fuse()
        model.to(self.device)
        return model
    
    def results(self, frame):
        return self.model.predict(frame, classes=[0], conf=0.3, verbose=False)[0]
    
    def get_results(self, results, frame):
        res_array = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.threshold:
                bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                res_array.append((bbox, float(score), int(class_id)))

        
        tracks = self.tracker.update_tracks(raw_detections=res_array, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue
                
            bboxes = track.to_ltrb()
            idx = track.track_id
            class_id = track.get_det_class()

            results.append((bboxes, idx, class_id))
    
        return results
    

    def build_groups(self, results):
        centers = []
        bboxes = []

        for (bbox, _, _) in results:
            x1, y1, x2, y2 = map(int, bbox)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            centers.append((cx, cy))
            bboxes.append((x1, y1, x2, y2))

        if len(centers) == 0:
            return {}
        
        centers = np.array(centers)

        clustering = DBSCAN(eps=120, min_samples=2).fit(centers)
        labels = clustering.labels_

        # labels is an array like [0, 0, 0, 1, 1, -1] - where:
        # 0 → cluster 0
        # 1 → cluster 1
        # -1 - noise → is not part of any group


        groups = {}

        for i, label in enumerate(labels):
            if label == -1:
                continue
            
            if label not in groups:
                groups[label] = []

            groups[label].append(bboxes[i])

        return groups

    def predict_action(self, frames):
        frames = np.stack(frames)

        flows = self.optical_flow.compute_dense_optical_flow(frames)

        flow_seq = self.optical_flow.extract_flow_features(flows)

        x = flow_seq.unsqueeze(0).to(self.device)

        pred = model.predict(x)

        res = self.tactics[int(pred)]

        proba = model.predict_proba(x)

        return res, proba

    def predict_action_thread(self, g_id, frames):
        action, proba = self.predict_action(frames)
        self.group_actions[g_id] = action
        self.action_proba[g_id] = proba
    
    def estimate_distance(self, groups):
        distcances = {}
        for g_id, bboxes in groups.items():
            distcances[g_id] = []
            for bbox in bboxes:
                dist = self.focal_length.estimate(bbox)
                distcances[g_id].append(dist)

        for g_id, d in distcances.items():
            d = np.array(d)
            self.group_dist[g_id] = int(np.mean(d))


    def estimate_threat(self, groups):
        for g_id, bbox in groups.items():
            size = len(bbox)
            dist = self.group_dist.get(g_id)
            action = self.group_actions.get(g_id)

            if action is None or dist is None:
                continue

            threat = self.threat_estimator.estimate(dist, size, action)
            self.group_threat[g_id] = threat


    def get_crop(self, frame, groups):
        crops = {}

        h, w, _ = frame.shape

        for g_id, group in groups.items():
            x1_list, y1_list, x2_list, y2_list = [], [], [], []

            for (x1, y1, x2, y2) in group:

                x1_list.append(x1)
                y1_list.append(y1)
                x2_list.append(x2)
                y2_list.append(y2)

            x1_group = max(0, min(x1_list) - self.group_box_const)
            y1_group = max(0, min(y1_list) - self.group_box_const)

            x2_group = min(w, max(x2_list) + self.group_box_const)
            y2_group = min(h, max(y2_list) + self.group_box_const)

            crop = frame[y1_group:y2_group, x1_group:x2_group]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (128, 128))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crops[g_id] = crop

        return crops

    
    def draw_groups(self, groups, frame):
        h, w, _ = frame.shape

        for g_id, group in groups.items():
            x1_list, y1_list, x2_list, y2_list = [], [], [], []

            for (x1, y1, x2, y2) in group:

                x1_list.append(x1)
                y1_list.append(y1)
                x2_list.append(x2)
                y2_list.append(y2)

            x1_group = max(0, min(x1_list) - self.group_box_const)
            y1_group = max(0, min(y1_list) - self.group_box_const)

            x2_group = min(w, max(x2_list) + self.group_box_const)
            y2_group = min(h, max(y2_list) + self.group_box_const)

            cv2.rectangle(frame, (x1_group, y1_group), (x2_group, y2_group), (0,0,255), 2)

            dist = self.group_dist[g_id]
            threat = self.group_threat.get(g_id, 'Analyzing...')
            size = len(group)
            
            # cv2.putText(frame, str(dist), (x1_group,y1_group), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)


            cx = (x1_group + x2_group) // 2
            cy = (y1_group + y2_group) // 2

            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

            action = self.group_actions.get(g_id, "analyzing")
            action_proba = self.action_proba.get(g_id, 0)
            
            

            text = f'{dist} m. | {action} ({action_proba:.2f})| {threat} | {size}'

            cv2.putText(frame, text, (x1_group + 50 ,y1_group), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2)


    def draw(self, results, frame):
        h, w, _ = frame.shape
        if results is not None:
            x1_list, y1_list, x2_list, y2_list = [], [], [], []
            colour = (0, 255, 0)
            for (bboxes, idx, class_id) in results:
                x1, y1, x2, y2 = map(int, bboxes)

                x1_list.append(x1)
                y1_list.append(y1)
                x2_list.append(x2)
                y2_list.append(y2)

                name = self.names[int(class_id)]
                text = f"{idx}:{name}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

            # if len(x1_list) > 0:

            #     x1_group = max(0, min(x1_list)-self.group_box_const)
            #     y1_group = max(0, min(y1_list)-self.group_box_const)

            #     x2_group = min(w, max(x2_list)+self.group_box_const)
            #     y2_group = min(h, max(y2_list)+self.group_box_const)

            #     cv2.rectangle(frame, (x1_group, y1_group), (x2_group, y2_group), colour, 2)
                
            #     x_group_center = (x1_group + x2_group) // 2
            #     y_group_center = (y1_group + y2_group) // 2

            #     cv2.circle(frame, (x_group_center, y_group_center), 4, colour, -1)

        return frame
        
    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()

            if not ret: 
                break

            results = self.results(frame)
            res_array = self.get_results(results, frame)

            for bbox, idx, class_id in res_array:

                x1, y1, x2, y2 = map(int, bbox)
                D = self.focal_length.estimate(bbox)
                cv2.putText(frame, f'{D} m', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            groups = self.build_groups(res_array)

            groups = self.group_tracker.track_groups(groups)

            crops = self.get_crop(frame, groups)
             
            for g_id, crop in crops.items():
                self.group_buffers[g_id].append(crop)

                if len(self.group_buffers[g_id]) == self.frames_needed:
                    frames = list(self.group_buffers[g_id])
                    threading.Thread(target=self.predict_action_thread, args=(g_id, frames)).start()



            self.estimate_distance(groups)

            self.estimate_threat(groups)

            # print(groups)

            self.draw_groups(groups, frame)
            upd_frmae = self.draw(res_array, frame)

            

            cv2.imshow('YOLO Tracker', upd_frmae)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            
        cap.release()
        cv2.destroyAllWindows()



path = 'videos/1.mp4'
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
yolo_link = "yolo12l.pt"
tracker = Tracker(path, device, yolo_link)
tracker()


