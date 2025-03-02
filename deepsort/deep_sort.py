from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import numpy as np
from ultralytics import YOLO
import cv2
import os

class DeepDetector:
    def __init__(self, path, device, threshold):
        self.path = path
        self.device = device
        self.model = self.load_model()
        self.names = self.model.names
        self.tracker = DeepSort(max_iou_distance=0.5, max_age=8)
        self.threshold = threshold

    def load_model(self):
        model = YOLO("models/detector.pt")
        model.to(self.device)
        model.fuse()
        return model

    def results(self, frame):
        results = self.model(frame)[0]
        return results

    def get_results(self, frame, results):
        print("d")
        res_array = []
        print(results.boxes.data.tolist())

        if len(results) != 0:
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > self.threshold:
                    res_array.append(([int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1)], float(score), int(class_id)))

                tracks = self.tracker.update_tracks(raw_detections=res_array, frame=frame)
                detected_objects = []

                if tracks:
                    for track in tracks:
                        bbox = track.to_ltrb()
                        idx = track.track_id
                        class_id = track.get_det_class()

                        detected_objects.append((bbox, idx, score, class_id))

                return frame, detected_objects
        else:
            return frame, None


    def detect_moving(self, frame, detected_objects, previous_positions, movings, moving_status = "Doesn't move"):
        if detected_objects:
            print("bhh")
            for obj in detected_objects:
                bbox, idx, score, class_id = obj

                x1, y1, x2, y2 = map(int, bbox)
                center = np.array([(x1+x2)/2, (y1+y2)/2])


                if idx in previous_positions:
                    prev_center = previous_positions[idx]
                    speed = np.linalg.norm(prev_center - center)
                    print(speed)

                    if speed > 0.0:
                        moving_status = "Moving"

                previous_positions[idx] = center
                movings[idx] = moving_status

                text = f"{idx}:{self.names[int(class_id)]}:{movings[idx]}:{score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return frame

        else:
            return frame



    def __call__(self):

        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        previous_positions = {}
        while True:
            movings = {}

            ret, frame = cap.read()
            results = self.results(frame)
            # print("X")
            frame, detected_objects = self.get_results(frame, results)
            upd_frame = self.detect_moving(frame, detected_objects, previous_positions, movings)

            cv2.imshow('Detection', upd_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


path = "videos/test_video_5.mp4"
# path = 1
device = "mps" if torch.backends.mps.is_available() else "cpu"
threshold = 0.45

tracker = DeepDetector(path, device, threshold)
tracker()
